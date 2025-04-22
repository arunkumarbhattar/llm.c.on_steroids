/*
This file trains the GPT-2 model.
This version is the clean, minimal, reference. As such:
- it runs on CPU.
- it does not make the code too complex; it is readable.
- it does not use any processor-specific instructions, intrinsics and such.
- it _does_ use a few OpenMP pragmas because this is a large speedup at very low cost
There will be other versions of this code that specialize it and make it fast.
*/

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#ifdef OMP
#include <omp.h>
#endif
// our own utilities
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#include "llmc/utils.h"
// defines: tokenizer_init, tokenizer_decode, tokenizer_free
#include "llmc/tokenizer.h"
// defines: dataloader_init, dataloader_reset, dataloader_next_batch, dataloader_free
#include "llmc/dataloader.h"


/*
 * Core Architecture: GPT-2 decoder stack with layers consisting of multi-head self-atention
 * followed by feed-forward MLP
 *
 * Training Process is as follows -->
 *
 * Initialization:
 *  Load a pretrainted GPT-2checkpoint (gpt2_build_from_checkpoint)
 *  ALlocate memory and intializes DOra parameters
 *  adapt hte laoded MHA weights and biases from checkpoint into GQA format ( avg KV heads).
 *  Calculate initial dora magnitude vector (m) based on adapted weights
 *  Set up data loaders for training and validation datasets and a tokenizer..
 *
 * Forward Pass (gpt2-forward) -->
 * take input token sequences and embeddings
 * Compute DORA effective weight for QKV, attention projection and MLP layer by calculating delta (B.A),
 *  adding it base weight W0, normalizing the result and scaling by the magnitude m --> all by using cached intermediate values
 *
 *  Execute GQA aware attention mechanism
 *  Apply resiual connections and feed-forward network using DORA adjusted weights
 *  Apply final layer normalization
 *  compute logits by multiplying with embedding matrix
 *  Calculate probabilities using softmax
 */

// ----------------------------------------------------------------------------
// all the individual layers' forward and backward passes
// B = batch_size, T = sequence_length, C = channels, V = vocab_size

// COmputes initial input embeddedings for the transofmer by combining token embeddings and positional embeddings
/*
 * Training (forward pass) --> prepares intial represetnations before passing it into transformer blocks
 * Infernece/Generation pass -->
 *
 */
void encoder_forward(float* out,
                   int* inp, float* wte, float* wpe,
                   int B, int T, int C) {
    // out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
    // inp is (B,T) of integers, holding the token ids at each (b,t) position
    // wte is (V,C) of token embeddings, short for "weight token embeddings"
    // wpe is (maxT,C) of position embeddings, short for "weight positional embedding"
    // loop through each token in the input batch (b = 0 to B)
    for (int b = 0; b < B; b++) {
        // and all sequence positions
        for (int t = 0; t < T; t++) {
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            // get the index of the token at inp[b, t]
            int ix = inp[b * T + t];
            // seek to the position in wte corresponding to the token
            float* wte_ix = wte + ix * C;
            // seek to the position in wpe corresponding to the position
            float* wpe_t = wpe + t * C;
            // add the two vectors and store the result in out[b,t,:]
            for (int i = 0; i < C; i++) {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}

/* Performs backward pass for embedding layer --> by calculating gradients of the lsos function
 * with respect to the token embedding weights (wte) and positional embedding weights (wpe)
 *
 * Receives upstream gradient --> takes dout as input. dout tensor contains the gradients flowing back from the
 * next layer in the network
 *
 * Iterate through each position (b,t) in batch
 *
 * For each position (b,t) --> identify the specific token index --> ix = inp[b * T + t]
 * and speciifc position index used t
 *
 * get specific gradient --> by accessing incoming gradient vector dout_bt corresponding to output embedding at (b,t)
 *
 * SInce forward operation was simple addition , gradient flowing back through an addition oepration is simply copied to both inputs.
 *
 * Why accumulate? --> specific token "the" (say) may appear multiple times in the batch at differnt positions. Similarly, a specific position is used for
 * all sequences in the batch.
 */
void encoder_backward(float* dwte, float* dwpe,
                      float* dout, int* inp,
                      int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * C + t * C;
            int ix = inp[b * T + t];
            float* dwte_ix = dwte + ix * C;
            float* dwpe_t = dwpe + t * C;
            for (int i = 0; i < C; i++) {
                float d = dout_bt[i];
                dwte_ix[i] += d;
                dwpe_t[i] += d;
            }
        }
    }
}

/*
 *  Layer normalization --> technique to stabilize training of neural networks
 *
 *  Normalizes activations across a single layer across the feature dimension (C)
 *
 *  For each input vector --> inp[b,t,:] --> this is what is happening
 *
 *  Calculate mean --> Compute mean of all elements within the vector (x) --> by summing across the C-dimension and dividing by C
 *
 *  Then calculate variance
 *
 *  Calculate reciprocal standard deviation
 *
 *  Then normalize by subtracting mean and diving by standard deviation.
 *
 *  Scale and shift --> apply leanred, element-wie affine transformation parameters
 *
 *  Store output --> The final scaled and shifted vector o in corresponding poisiton in the output tensor out
 *
 *  Cache for backward pass --> calculated mean and reciprocal standward deviation for each vector x are stored in the mean and rstd
 *  tensors respectively --> these are needed later for recalculating gradients effectively during backward pass
 *
 * In which parts of the pass are these seen -->
 *  Before attention , FFN, and
 Final Lyaer norm */
void layernorm_forward(float* out, float* mean, float* rstd,
                       float* inp, float* weight, float* bias,
                       int B, int T, int C) {
    // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    // both inp and out are (B,T,C) of the activations
    // mean and rstd are (B,T) buffers, to be used later in backward pass
    // at each position (b,t) of the input, the C-dimensional vector
    // of activations gets normalized, then scaled and shifted
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            float* x = inp + b * T * C + t * C;
            // calculate the mean
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd (reciprocal standard deviation)
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m)); // normalize
                float o = n * weight[i] + bias[i]; // scale and shift
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

/* Computes gradients for the LN oepration performed in the layernorm_forward
 *
 * How loss function changes wrt input to layerNorm(dinp)
 * The learnable scale parameter (dweight)
 * The learnable shift parameter(dbias)
 *
 * uses chain rule to propagate the incoming gradient dout --> backward thorugh the LayerNorm calculation
 *
 */
void layernorm_backward(float* dinp, float* dweight, float* dbias,
                        float* dout, float* inp, float* weight, float* mean, float* rstd,
                        int B, int T, int C) {
    //loop through each vector position
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            //get the incoming gradient
            float* dout_bt = dout + b * T * C + t * C;
            //original input
            float* inp_bt = inp + b * T * C + t * C;
            float* dinp_bt = dinp + b * T * C + t * C;
            //cached mean_bt and rstd_bt from the forward pass
            float mean_bt = mean[b * T + t];
            float rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            //here we calculate the sums needed for input gradient (dinp)
            float dnorm_mean = 0.0f;
            float dnorm_norm_mean = 0.0f;
            for (int i = 0; i < C; i++) {
                //compupte the normalized input using cached values
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                //compute gradient after shift but before normalization and scaling
                float dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }

            //this captures how the overall gradeint interacts with the normalization statistics
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // now iterate again and accumulate all the gradients
            // second pass --> gradient calculation and accumulation
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                float dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_mean; // term 2
                dval -= norm_bti * dnorm_norm_mean; // term 3
                dval *= rstd_bt; // final scale
                dinp_bt[i] += dval;
            }
        }
    }
}

/*
 * Standard matrix multiplication operation used for linear transformations in neural networks -->
 * Applied across a batch (B) and sequence (T) dimension
 * inp -> Input (B, T, C)
 * weight --> Weight (OC,C) --> weight matrix of the linear layer
 * bias --> Bias (OC) --> optional bias vector
 * out --> Output (B, T, OC) --> result of transformation
 *
 * OC  --> output dimension
 *
 *
 *
 */
void matmul_forward_naive(float* out,
                         const float* inp, const float* weight, const float* bias,
                         int B, int T, int C, int OC) {
    // the most naive implementation of matrix multiplication
    // this serves as an algorithmic reference, and as a fallback for
    // unfriendly input shapes inside matmul_forward(), below.
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int bt = b * T + t;
            for (int o = 0; o < OC; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                for (int i = 0; i < C; i++) {
                    val += inp[bt * C + i] * weight[o*C + i];
                }
                out[bt * OC + o] = val;
            }
        }
    }
}

//restructured and OpenMP optimized matrix multiplication
void matmul_forward(float* out,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
    // most of the running time is spent here and in matmul_backward
    // therefore, the implementation below is very mildly optimized
    // this function is otherwise identical to that of matmul_forward_naive()
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    // make sure the tiled loop will be correct or fallback to naive version
    const int LOOP_UNROLL = 8;
    if (B*T % LOOP_UNROLL != 0) {
        matmul_forward_naive(out, inp, weight, bias, B, T, C, OC);
        return;
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    #pragma omp parallel for
    for (int obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        for (int o = 0; o < OC; o++) {
            // we'll keep LOOP_UNROLL many results in registers
            float result[LOOP_UNROLL];
            // initialize the bias, if it exists
            for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                result[ibt] = (bias != NULL) ? bias[o] : 0.0f;
            }
            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into FMAs
            for (int i = 0; i < C; i++) {
                float w = weight[i + o * C];
                for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                    int bt = obt + ibt;
                    result[ibt] += inp[bt * C + i] * w;
                }
            }
            // write back results to main memory
            for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                int bt = obt + ibt;
                out[bt * OC + o] = result[ibt];
            }
        }
    }
}

/*
 * Here we calculate the gradients for the matrix multiplication operation performed in matmul_forward
 * Basically, how much should original inputs to the matrix multiplication should change to minimize the loss
 *
 * Specifically, we calculate -->
 *              dinp --> gradient of the loss with respect to input activations
 *              dweight --> gradient of the loss with respect to the weight matrix
 *              dbias --> gradient of the loss with respect to the bias vector
 *
 * Calculating garduetns (dinp)
 *
 */
void matmul_backward(float* dinp, float* dweight, float* dbias,
                     const float* dout, const float* inp, const float* weight,
                     int B, int T, int C, int OC) {

    if (dout == NULL) {
        fprintf(stderr, "FATAL: dout cannot be NULL in matmul_backward.\n");
        exit(EXIT_FAILURE);
    }
    if (dinp != NULL && weight == NULL) {
         fprintf(stderr, "FATAL: weight cannot be NULL if dinp is requested in matmul_backward.\n");
         exit(EXIT_FAILURE);
    }
     if ((dweight != NULL || dbias != NULL) && inp == NULL) {
         fprintf(stderr, "FATAL: inp cannot be NULL if dweight or dbias is requested in matmul_backward.\n");
         exit(EXIT_FAILURE);
     }

    // calculate gradient w.r.t Input X
    if (dinp != NULL) {
        //only execute if the caller requested the input gardient
        //parallelize the lops over batch B and time T
        #pragma omp parallel for collapse(2)
        for (int b = 0; b < B; b++) { //loop through each sequence
            for (int t = 0; t < T; t++) { //Loop through each token (time step)

                //reference to upstream gradent vector for this position (b,t)
                const float* dout_bt = dout + b * T * OC + t * OC;
                //pointer to where the calculate input gradient for this position

                float* dinp_bt = dinp + b * T * C + t * C;

                //Iterate thorugh the output channel
                for (int o = 0; o < OC; o++) {
                    //pointer to o-th row of weight matrix
                    const float* wrow = weight + o*C;
                    //upstream gradeint value for the oth otput channel
                    float d = dout_bt[o];
                    //accumulate gradient across token vector
                    for (int i = 0; i < C; i++) {
                        dinp_bt[i] += wrow[i] * d;
                    }
                }
            }
        }
    }


    //calculate gradients with respect to weight and Bias B
    if (dweight != NULL || dbias != NULL) {
        #pragma omp parallel for
        for (int o = 0; o < OC; o++) {

            for (int b = 0; b < B; b++) {
                for (int t = 0; t < T; t++) {
                    //pointer to the upstream gradient vector for this position (b,t)
                    const float* dout_bt = dout + b * T * OC + t * OC;
                    //pointer to the original input vector for this position (b,t)
                    const float* inp_bt = inp + b * T * C + t * C;
                    //upstream gradeint value for the current output channel
                    float d = dout_bt[o];

                    //calculate gradient for bias
                    if (dbias != NULL) {
                        dbias[o] += d;
                    }
                    //calculate  gradient for weight
                    if (dweight != NULL) {
                        // pointer to the o-th row of the weight gradient matrix
                         float* dwrow = dweight + o*C;
                         //iterate through each input channel
                         for (int i = 0; i < C; i++) {
                             dwrow[i] += inp_bt[i] * d;
                         }
                    }
                }
            }
        }
    }
}

/*
 * Here we calculate self-atention for a sequence of input vectors using GAA
 * Basically --> each token (at pos t) looks back to previous tokens from positions (0 to t0 and decide
 * which ones are most relevant to understand its own meaning in context.
 *
 * Input --> receives pre-computed  Qeury, Key and Value vectors packed into inp tensor.
 *
 * In GQA --> many Query heads (NH) and smaller shared set of key and value heads (NKVH).
 * Multiple Query heads form a group and share the same K and V heads
 *
 * Now --> processing happens only per Query head -->
 *     --> For each query head  at each position t
 *         --> Find shared K/V --> which group the query head belongs to and identify corresponding shared Key and valye heads (kv_head_index)
 *         --> Calculate Scores --> Compute attention score by taking scaled dot product between the current Query vector (query_t) and key vectors (key_t2) from
 *                                  shared K head for all positions up to t (t2 <=t). Scaling factor 1 /sqrt(HeadSize) helps stabilize gradients
 *         --> Softmax --> apply softmax function to these scores (t2 <= t) to turn them into attention weights
 *         --> Weighted Sum --> calculates the weighted sum of value vectors (value_t2) from the shared V head.
 *
 *     --> Output --> final output vector for posiiton t is formed by concatenating the outputs from all the individual query heads (NH)
 *     --> function also stores the intermediate pre-softmax scores (preat) and the final attention weights (att).
 *
 *
 */
void attention_forward(float* out, // Output tensor (B, T, C)
                       float* preatt, // Buffer to store pre-softmax attention scores
                       float* att, //Buffer to store post-softmax attention weights
                       float* inp, //Input tensor containing packed Q, K and V projections
                       int B, // Batch size
                       int T, // Sequence length
                       int C, // Channels
                       int NH, // Number of query Heads
                       int NKVH //Number of Key/Value heads
                       ) // Added NKVH
{
    //head size
    int HS = C / NH;
    assert(NH > 0); //atleast one query head
    assert(NKVH > 0); //atleast one K/V head
    assert(NH % NKVH == 0); // Number of query heads must be divisible by number of KV heads

    //how many query heads share one KV head
    int q_heads_per_kv_head = NH / NKVH;
    size_t q_dim = NH * HS; // = C --> total dimension for all query vectors
    size_t kv_dim = NKVH * HS;

    // Offset between consecutive token representations in the 'inp' tensor
    //Layout looks like [Q0....Q(NH-1), K0....K(NKVH-1), V0...V(NKVH-1)]
    size_t qkv_layer_offset = q_dim + 2 * kv_dim;
    //scaling factor for dot product
    float scale = 1.0f / sqrtf((float)HS); // Cast HS to float for sqrtf

    #pragma omp parallel for
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) { // Loop over each query head
                // Pointer to query vector for head 'h' at position (b,t) within 'inp'
                float* query_t = inp + b * T * qkv_layer_offset + t * qkv_layer_offset + h * HS;
                // Pointers for pre-attention and attention scores for this (b,t)
                float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;

                // index of this key/value head group that this head consumes
                int kv_head_index = h / q_heads_per_kv_head;

                // Base pointers for the shared K and V heads for this group within the inp tensor
                float* key_base = inp + b * T * qkv_layer_offset + q_dim + kv_head_index * HS;
                float* value_base = inp + b * T * qkv_layer_offset + q_dim + kv_dim + kv_head_index * HS;

                // ---> Pass 1: Calculate query dot key scores and maxval ---
                float maxval = -10000.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    //key vector for timestep t2
                    float* key_t2 = key_base + t2 * qkv_layer_offset;

                    // query_t dot key_t2
                    float val = 0.0f;
                    for (int i = 0; i < HS; i++) { // Serial loop over head size
                        val += query_t[i] * key_t2[i];
                    }

                    val *= scale;
                    if (val > maxval) {
                        maxval = val;
                    }
                    preatt_bth[t2] = val; // store pre-softmax attention score
                }

                // --> Pass 2 Calculate exponentials for softmax
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv; // att[b, h, t, t2] = exp(score - maxval)
                }

                // ---> Pass 3: Normalize softmax scores
                float expsum_inv = (expsum == 0.0f || !isfinite(expsum)) ? 0.0f : 1.0f / expsum;
                for (int t2 = 0; t2 <= t; t2++) {
                    //normalize by multiplying with inverse sum
                    att_bth[t2] *= expsum_inv;
                }

                 for (int t2 = t + 1; t2 < T; t2++) {
                    att_bth[t2] = 0.0f;
                 }

                // ---> Pass 4: Calculate the weighted sum of Value vectors
                // Output pointer for this specific query head h
                float* out_bth = out + b * T * C + t * C + h * HS;
                // Initialize output vector for this head to zeros
                for (int i = 0; i < HS; i++) { out_bth[i] = 0.0f; }

                // Weighted sum of values based on attention weights
                for (int t2 = 0; t2 <= t; t2++) {
                    float* value_t2 = value_base + t2 * qkv_layer_offset;
                    float att_btht2 = att_bth[t2];
                    for (int i = 0; i < HS; i++) { // Serial loop over head size
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
}

/*
 * Basically gotta figure out how much should the Query (Q), Key (K) and Value(V) vectors  need to change to reduce overall loss,
 * based on the error signal (dout) coming from previous layer in the backward pass
 *
 * dout --> tells us how wrong the final output of the attention layer was
 *
 * Blame value and attention weights --> Figure out how much the value (V) vectors and attention weights (A)
 *  contributed to this output (dout)
 *
 *      So gradient wrt V (dvalue_t2)
 *      and gradient wrt attention weights (datt_bth)
 *
 * Blame pre-softmax scores --> then we gotta propagate the blame from the attention weights  (datt_bth) back
 *      through the Softmax function to figure out how much the pre-softmax scores (S) should change.
 *
 * Then blame Query and Key --> Propagate blame from pre-softmax scores (dpreatt_bth) back through scaled dot product operation
 *  to figure out how much the original Query (Q) vectors and Key (K) vectors should change
 *
 *  GQA accumulation --> Because multiple query heads share the same K and V vectors in GQA within a group --> gradients
 *  calculated for these shared K and V vectors are accumulated across all query heads within the same group
 *
 *  the function fills the dinp buffer with these calculated gradients. These gradients can be further propagated backward
 *  to the linear layer that originally produced Q, K and V
 *
 */
void attention_backward(float* dinp, // out: grad wrt projected QKV (B, T, (NH+2*NKVH)*HS)
                        float* dpreatt, float* datt, // scratch
                        float* dout, // in: grad wrt output (B, T, C)
                        float* inp, // in: projected QKV from forward (B, T, (NH+2*NKVH)*HS)
                        float* att, // in: attention probabilities (B, NH, T, T)
                        int B, int T, int C, int NH, int NKVH) // Added NKVH
{
    int HS = C / NH;
    assert(NH > 0);
    assert(NKVH > 0);
    assert(NH % NKVH == 0);
    int q_heads_per_kv_head = NH / NKVH;
    size_t q_dim = NH * HS;
    size_t kv_dim = NKVH * HS;

    // size of the QKV projection per token based on GQA dimensions
    size_t qkv_layer_offset = q_dim + 2 * kv_dim;
    float scale = 1.0f / sqrtf((float)HS);
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++)
            {
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;
                // Pointer to scratch space for gradient w.r.t attention weights
                float* datt_bth = datt + b*NH*T*T + h*T*T + t*T;
                float* dpreatt_bth = dpreatt + b*NH*T*T + h*T*T + t*T;

                // KV head group index for this query head
                int kv_head_index = h / q_heads_per_kv_head;

                // Pointers for the QKV tensor (forward activations)
                float* query_t = inp + b * T * qkv_layer_offset + t * qkv_layer_offset + h * HS;
                float* key_base = inp + b * T * qkv_layer_offset + q_dim + kv_head_index * HS;
                float* value_base = inp + b * T * qkv_layer_offset + q_dim + kv_dim + kv_head_index * HS;

                // Pointers for the OUTPUT QKV gradients (dinp)
                float* dquery_t = dinp + b * T * qkv_layer_offset + t * qkv_layer_offset + h * HS;
                float* dkey_base = dinp + b * T * qkv_layer_offset + q_dim + kv_head_index * HS;
                float* dvalue_base = dinp + b * T * qkv_layer_offset + q_dim + kv_dim + kv_head_index * HS;

                // Pointer to incoming gradient that is wrt output of this specific head
                float* dout_bth = dout + b * T * C + t * C + h * HS;

                // ---> Backward pass 4 -- Backprop thorugh weighted sum
                // Calculate gradient wrt attention weights and wrt Values
                for(int t2=0; t2<T; ++t2) { datt_bth[t2] = 0.0f; }

                for (int t2 = 0; t2 <= t; t2++) {
                    float* value_t2 = value_base + t2 * qkv_layer_offset;
                    float* dvalue_t2 = dvalue_base + t2 * qkv_layer_offset;
                    float att_weight = att_bth[t2];
                    for (int i = 0; i < HS; i++) {
                        datt_bth[t2] += value_t2[i] * dout_bth[i];
                        dvalue_t2[i] += att_weight * dout_bth[i];
                    }
                }

                // ---> Backward pass 2 & 3: Backprop through Softmasx ---
                // Calculates gradient w.r.t. pre-softmax scores (dpreatt_bth) using datt_bth (post softmax)
                for(int t3=0; t3<T; ++t3) { dpreatt_bth[t3] = 0.0f; }

                for (int t2 = 0; t2 <= t; t2++) {
                    for (int t3 = 0; t3 <= t; t3++) {
                        float att_t2 = att_bth[t2];
                        float att_t3 = att_bth[t3];
                        float indicator = (t2 == t3) ? 1.0f : 0.0f;
                        float local_derivative = att_t2 * (indicator - att_t3);
                        dpreatt_bth[t3] += scale * local_derivative * datt_bth[t2];
                    }
                }

                // ---> Backward pass 1: Matmul backward (dQ, dK) ---
                // Calculates gradient w.r.t. query (dquery_t) and key (dkey_t2) using dpreatt_bth
                for (int t2 = 0; t2 <= t; t2++) {
                    float* key_t2 = key_base + t2 * qkv_layer_offset; // Shared K/V head value
                    float* dkey_t2 = dkey_base + t2 * qkv_layer_offset; // Shared K/V head gradient
                    float dpreatt_val = dpreatt_bth[t2]; // Gradient w.r.t presoftmax score calculated above
                    for (int i = 0; i < HS; i++) {
                        dquery_t[i] += key_t2[i] * dpreatt_val;
                        dkey_t2[i] += query_t[i] * dpreatt_val;
                    }
                }
            }
        }
    }
}

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
void gelu_forward(float* out, float* inp, int N) {
    // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
    }
}

// we want to use -Ofast optimization, but sadly GeLU breaks, so disable this flag just for it (#168)
#pragma float_control(precise, on, push)
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-finite-math-only")))
#endif
void gelu_backward(float* dinp, float* inp, float* dout, int N) {
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] += local_grad * dout[i];
    }
}
#pragma float_control(pop)

void residual_forward(float* out, float* inp1, float* inp2, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}

void residual_backward(float* dinp1, float* dinp2, float* dout, int N) {
    for (int i = 0; i < N; i++) {
        dinp1[i] += dout[i];
        dinp2[i] += dout[i];
    }
}

void softmax_forward(float* probs, float* logits, int B, int T, int V, int Vp) {
    // output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t position)
    // input: logits is (B,T,Vp) of the unnormalized log probabilities
    // Vp is the padded vocab size (for efficiency), V is the "real" vocab size
    // example: Vp is 50304 and V is 50257
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // probs <- softmax(logits)
            float* logits_bt = logits + b * T * Vp + t * Vp;
            float* probs_bt = probs + b * T * Vp + t * Vp;

            // maxval is only calculated and subtracted for numerical stability
            float maxval = -10000.0f; // TODO something better
            for (int i = 0; i < V; i++) {
                if (logits_bt[i] > maxval) {
                    maxval = logits_bt[i];
                }
            }
            float sum = 0.0f;
            for (int i = 0; i < V; i++) {
                probs_bt[i] = expf(logits_bt[i] - maxval);
                sum += probs_bt[i];
            }
            // note we only loop to V, leaving the padded dimensions
            for (int i = 0; i < V; i++) {
                probs_bt[i] /= sum;
            }
            // for extra super safety we may wish to include this too,
            // forcing the probabilities here to be zero, but it shouldn't matter
            for (int i = V; i < Vp; i++) {
                probs_bt[i] = 0.0f;
            }
        }
    }
}

/*
 * Cross-entropy is a standard way to measure how well the mode's predicted probability matches the single correct
 * target token.
 *
 * probs: probs[b, t, i] is the predicted probability of token i at position (b,t)
 *
 * targets: the tensor containing the correct integer token IDs for each position and shape (b,t)
 *
 * output --> A tensor of shape (B, t) where losses[b, t] will store calculated cross
 * entropy loss for the prediction at position (b, t)
 *
 * softmax_forard function generates the full probability distribution (probs_bt) over entire vocabulary (V) for current pos (b, t)
 * This vector basically contains the mode;s belief about the likelihood of every possible next token.
 *
 *
 * Ground Truth --> the targets array holds the ground truth --> integer index of a single token that actually comes next at position (b, t).
 * retrieved using -> int ix = targets[b * T + t]
 *
 * if probs_bt[ix] was high -> model did very well for the next token and is largely right --> -logf() is low
 *              and vice versa
 *
 *
 */
void crossentropy_forward(float* losses,
                          float* probs, int* targets,
                          int B, int T, int Vp) {
    // output: losses is (B,T) of the individual losses at each position
    // input: probs are (B,T,Vp) of the probabilities
    // input: targets is (B,T) of integers giving the correct index in logits
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // loss = -log(probs[target])
            float* probs_bt = probs + b * T * Vp + t * Vp;
            int ix = targets[b * T + t];
            losses[b * T + t] = -logf(probs_bt[ix]);
        }
    }
}

/*
 * Goal of this is to calculate the gradient of the overall loss function with
 * respect to the logits (raw scores that were fed into the softmax function).
 *
 *
 */
void crossentropy_softmax_backward(float* dlogits,
                           float* dlosses, float* probs, int* targets,
                           int B, int T, int V, int Vp) {
    // backwards through both softmax and crossentropy
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dlogits_bt = dlogits + b * T * Vp + t * Vp;
            float* probs_bt = probs + b * T * Vp + t * Vp;
            float dloss = dlosses[b * T + t];
            int ix = targets[b * T + t];
            // note we only loop to V, leaving the padded dimensions
            // of dlogits untouched, so gradient there stays at zero
            for (int i = 0; i < V; i++) {
                float p = probs_bt[i];
                float indicator = i == ix ? 1.0f : 0.0f;
                dlogits_bt[i] += (p - indicator) * dloss;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPT-2 model definition

typedef struct {
    int max_seq_len;
    int vocab_size;
    int padded_vocab_size;
    int num_layers;
    int num_heads; // Number of query heads (per group GQA)
    int num_kv_heads; // Number of KEY/VALUE heads (must divide num_heads)
    int channels;
} GPT2Config;

// the parameters of the model
#define NUM_PARAMETER_TENSORS 16
typedef struct {
    float* wte; // (V, C)
    float* wpe; // (maxT, C)
    float* ln1w; // (L, C)
    float* ln1b; // (L, C)
    float* qkvw; // (L, 3*C, C) - Base weights (Frozen)
    float* qkvb; // (L, 3*C) - Base biases (Trainable or Frozen)
    float* qkvb_gqa; // (L, OC_qkv_gqa) - Adapted GQA biases (TRAINABLE)
    float* qkvw_A;     // LoRA A matrices (L, r, C) -> Size: L*r*C
    float* qkvw_B;     // LoRA B matrices (L, 3*C, r) -> Size: L*3*C*r (Trainable)
    float* qkvw_m;     // Magnitude vectors (L, 3*C) -> Size: L*3*C (Trainable)
    float* attprojw_A; // (L, r, C)
    float* attprojw_B; // (L, C, r)
    float* attprojw_m; // (L, C)
    float* fcw_A;      // (L, r, C)
    float* fcw_B;      // (L, 4*C, r)
    float* fcw_m;      // (L, 4*C)
    float* fcprojw_A;  // (L, r, 4*C)
    float* fcprojw_B;  // (L, C, r)
    float* fcprojw_m;  // (L, C)
    float* attprojw; // (L, C, C)
    float* attprojb; // (L, C)
    float* ln2w; // (L, C)
    float* ln2b; // (L, C)
    float* fcw; // (L, 4*C, C)
    float* fcb; // (L, 4*C)
    float* fcprojw; // (L, C, 4*C)
    float* fcprojb; // (L, C)
    float* lnfw; // (C)
    float* lnfb; // (C)
} ParameterTensors;

// Corrected version of fill_in_parameter_sizes
void fill_in_parameter_sizes(size_t* param_sizes, GPT2Config config) {
    size_t Vp = config.padded_vocab_size;
    size_t C = config.channels;
    size_t maxT = config.max_seq_len;
    size_t L = config.num_layers;
    // Note: GQA specific dimensions are not needed here for loading MHA params.

    param_sizes[0] = Vp * C;            // wte
    param_sizes[1] = maxT * C;          // wpe
    param_sizes[2] = L * C;             // ln1w
    param_sizes[3] = L * C;             // ln1b
    param_sizes[4] = L * 3 * C * C;     // qkvw (Original MHA weight size)
    param_sizes[5] = L * 3 * C;         // qkvb (Original MHA bias size)
    param_sizes[6] = L * C * C;         // attprojw
    param_sizes[7] = L * C;             // attprojb
    param_sizes[8] = L * C;             // ln2w
    param_sizes[9] = L * C;             // ln2b
    param_sizes[10] = L * (4 * C) * C;  // fcw
    param_sizes[11] = L * (4 * C);      // fcb
    param_sizes[12] = L * C * (4 * C);  // fcprojw
    param_sizes[13] = L * C;            // fcprojb
    param_sizes[14] = C;                // lnfw
    param_sizes[15] = C;                // lnfb
}

float* malloc_and_point_parameters(ParameterTensors* params, size_t* param_sizes) {
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += param_sizes[i];
    }

    printf("Allocating %.2f MB for %d base parameter tensors...\n",
           num_parameters * sizeof(float) / (1024.0f * 1024.0f), NUM_PARAMETER_TENSORS);

    float* params_memory = (float*)mallocCheck(num_parameters * sizeof(float));

    float** ptrs[] = {
        &params->wte,      // Index 0
        &params->wpe,      // Index 1
        &params->ln1w,     // Index 2
        &params->ln1b,     // Index 3
        &params->qkvw,     // Index 4
        &params->qkvb,     // Index 5
        &params->attprojw, // Index 6
        &params->attprojb, // Index 7
        &params->ln2w,     // Index 8
        &params->ln2b,     // Index 9
        &params->fcw,      // Index 10
        &params->fcb,      // Index 11
        &params->fcprojw,  // Index 12
        &params->fcprojb,  // Index 13
        &params->lnfw,     // Index 14
        &params->lnfb      // Index 15
    };

    assert(sizeof(ptrs) / sizeof(ptrs[0]) == NUM_PARAMETER_TENSORS);

    float* params_memory_iterator = params_memory;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = params_memory_iterator;
        params_memory_iterator += param_sizes[i];
    }

    params->qkvb_gqa = NULL;
    params->qkvw_A = NULL;
    params->qkvw_B = NULL;
    params->qkvw_m = NULL;
    params->attprojw_A = NULL;
    params->attprojw_B = NULL;
    params->attprojw_m = NULL;
    params->fcw_A = NULL;
    params->fcw_B = NULL;
    params->fcw_m = NULL;
    params->fcprojw_A = NULL;
    params->fcprojw_B = NULL;
    params->fcprojw_m = NULL;

    return params_memory;
}

#define NUM_ACTIVATION_TENSORS (23 + 4*4)
typedef struct {
    float* encoded; // (B, T, C)
    float* ln1; // (L, B, T, C)
    float* ln1_mean; // (L, B, T)
    float* ln1_rstd; // (L, B, T)
    float* qkv; // (L, B, T, 3*C)
    float* atty; // (L, B, T, C)
    float* preatt; // (L, B, NH, T, T)
    float* att; // (L, B, NH, T, T)
    float* attproj; // (L, B, T, C)
    float* residual2; // (L, B, T, C)
    float* ln2; // (L, B, T, C)
    float* ln2_mean; // (L, B, T)
    float* ln2_rstd; // (L, B, T)
    float* fch; // (L, B, T, 4*C)
    float* fch_gelu; // (L, B, T, 4*C)
    float* fcproj; // (L, B, T, C)
    float* residual3; // (L, B, T, C)
    float* lnf; // (B, T, C)
    float* lnf_mean; // (B, T)
    float* lnf_rstd; // (B, T)
    float* logits; // (B, T, Vp)
    float* probs; // (B, T, Vp)
    float* losses; // (B, T)
    float* qkv_delta_cache;       // (L, 3*C, C)
    float* qkv_V_prime_cache;     // (L, 3*C, C)
    float* qkv_norm_V_prime_cache;// (L, 3*C)
    float* qkv_W_dora_cache;      // (L, 3*C, C)
    // AttProj
    float* attproj_delta_cache;   // (L, C, C)
    float* attproj_V_prime_cache; // (L, C, C)
    float* attproj_norm_V_prime_cache; // (L, C)
    float* attproj_W_dora_cache;  // (L, C, C)
    // FC
    float* fc_delta_cache;        // (L, 4*C, C)
    float* fc_V_prime_cache;      // (L, 4*C, C)
    float* fc_norm_V_prime_cache; // (L, 4*C)
    float* fc_W_dora_cache;       // (L, 4*C, C)
    // FCProj
    float* fcproj_delta_cache;    // (L, C, 4*C)
    float* fcproj_V_prime_cache;  // (L, C, 4*C)
    float* fcproj_norm_V_prime_cache; // (L, C)
    float* fcproj_W_dora_cache;   // (L, C, 4*C)
} ActivationTensors;

void fill_in_activation_sizes(size_t* act_sizes, GPT2Config config, int B, int T) {
    size_t C = config.channels;
    size_t NH = config.num_heads;
    size_t L = config.num_layers;
    size_t Vp = config.padded_vocab_size;
    size_t NKVH = config.num_kv_heads;  // Get from config
    size_t HS = C / NH;
    size_t qkv_out_dim = (NH + 2 * NKVH) * HS; // GQA output dimension for fused QKV
    size_t OC_fc = 4 * C;

    act_sizes[0] = B * T * C;         // encoded
    act_sizes[1] = L * B * T * C;     // ln1
    act_sizes[2] = L * B * T;         // ln1_mean
    act_sizes[3] = L * B * T;         // ln1_rstd
    act_sizes[4] = L * B * T * qkv_out_dim; // qkv
    act_sizes[5] = L * B * T * C;     // atty
    act_sizes[6] = L * B * NH * T * T; // preatt
    act_sizes[7] = L * B * NH * T * T; // att
    act_sizes[8] = L * B * T * C;     // attproj
    act_sizes[9] = L * B * T * C;     // residual2
    act_sizes[10] = L * B * T * C;    // ln2
    act_sizes[11] = L * B * T;        // ln2_mean
    act_sizes[12] = L * B * T;        // ln2_rstd
    act_sizes[13] = L * B * T * 4 * C; // fch
    act_sizes[14] = L * B * T * 4 * C; // fch_gelu
    act_sizes[15] = L * B * T * C;    // fcproj
    act_sizes[16] = L * B * T * C;    // residual3
    act_sizes[17] = B * T * C;        // lnf
    act_sizes[18] = B * T;            // lnf_mean
    act_sizes[19] = B * T;            // lnf_rstd
    act_sizes[20] = B * T * Vp;       // logits
    act_sizes[21] = B * T * Vp;       // probs
    act_sizes[22] = B * T;            // losses

    act_sizes[23] = L * qkv_out_dim * C; // qkv_delta_cache
    act_sizes[24] = L * qkv_out_dim * C; // qkv_V_prime_cache
    act_sizes[25] = L * qkv_out_dim;     // qkv_norm_V_prime_cache
    act_sizes[26] = L * qkv_out_dim * C; // qkv_W_dora_cache

    act_sizes[27] = L * C * C;      // attproj_delta_cache
    act_sizes[28] = L * C * C;      // attproj_V_prime_cache
    act_sizes[29] = L * C;          // attproj_norm_V_prime_cache
    act_sizes[30] = L * C * C;      // attproj_W_dora_cache

    act_sizes[31] = L * OC_fc * C;  // fc_delta_cache
    act_sizes[32] = L * OC_fc * C;  // fc_V_prime_cache
    act_sizes[33] = L * OC_fc;      // fc_norm_V_prime_cache
    act_sizes[34] = L * OC_fc * C;  // fc_W_dora_cache

    act_sizes[35] = L * C * OC_fc;  // fcproj_delta_cache
    act_sizes[36] = L * C * OC_fc;  // fcproj_V_prime_cache
    act_sizes[37] = L * C;          // fcproj_norm_V_prime_cache
    act_sizes[38] = L * C * OC_fc;  // fcproj_W_dora_cache
}

float* malloc_and_point_activations(ActivationTensors* acts, size_t* act_sizes) {
    size_t num_activations = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        num_activations += act_sizes[i];
    }
    printf("Allocating %.2f MB for activations (including ALL DORA caches)\n",
           num_activations * sizeof(float) / (1024.0f*1024.0f));
    float* acts_memory = (float*)mallocCheck(num_activations * sizeof(float));

    float** ptrs[] = {
        // Base Activations (indices 0-22)
        &acts->encoded, &acts->ln1, &acts->ln1_mean, &acts->ln1_rstd, &acts->qkv, &acts->atty,
        &acts->preatt, &acts->att, &acts->attproj, &acts->residual2, &acts->ln2, &acts->ln2_mean,
        &acts->ln2_rstd, &acts->fch, &acts->fch_gelu, &acts->fcproj, &acts->residual3, &acts->lnf,
        &acts->lnf_mean, &acts->lnf_rstd, &acts->logits, &acts->probs, &acts->losses,
        // QKV DORA Cache (indices 23-26)
        &acts->qkv_delta_cache, &acts->qkv_V_prime_cache, &acts->qkv_norm_V_prime_cache, &acts->qkv_W_dora_cache,
        // AttProj DORA Cache (indices 27-30)
        &acts->attproj_delta_cache, &acts->attproj_V_prime_cache, &acts->attproj_norm_V_prime_cache, &acts->attproj_W_dora_cache,
        // FC DORA Cache (indices 31-34)
        &acts->fc_delta_cache, &acts->fc_V_prime_cache, &acts->fc_norm_V_prime_cache, &acts->fc_W_dora_cache,
        // FCProj DORA Cache (indices 35-38)
        &acts->fcproj_delta_cache, &acts->fcproj_V_prime_cache, &acts->fcproj_norm_V_prime_cache, &acts->fcproj_W_dora_cache
    };
    assert(sizeof(ptrs) / sizeof(ptrs[0]) == NUM_ACTIVATION_TENSORS);

    float* acts_memory_iterator = acts_memory;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        *(ptrs[i]) = acts_memory_iterator;
        acts_memory_iterator += act_sizes[i];
    }
    return acts_memory;
}

typedef struct {
    GPT2Config config;
    // the weights (parameters) of the model, and their sizes
    ParameterTensors params;
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    float* params_memory;
    size_t num_parameters;
    // gradients of the weights
    ParameterTensors grads;
    float* grads_memory;
    // buffers for the AdamW optimizer
    float* m_memory;
    float* v_memory;
    // the activations of the model, and their sizes
    ActivationTensors acts;
    size_t act_sizes[NUM_ACTIVATION_TENSORS];
    float* acts_memory;
    size_t num_activations;
    // gradients of the activations
    ActivationTensors grads_acts;
    float* grads_acts_memory;
    // other run state configuration
    int batch_size; // the batch size (B) of current forward pass
    int seq_len; // the sequence length (T) of current forward pass
    int* inputs; // the input tokens for the current forward pass
    int* targets; // the target tokens for the current forward pass
    float mean_loss; // after a forward pass with targets, will be populated with the mean loss
    ParameterTensors grads_dora; // Gradients for DORA A, B, m parameters
    float* params_memory_dora;   // Memory block for ALL DORA A, B, m parameters
    float* grads_memory_dora;    // Memory block for ALL DORA A, B, m gradients
    float* m_memory_dora;        // Memory block for ALL DORA m_memory states
    float* v_memory_dora;        // Memory block for ALL DORA v_memory states
    size_t dora_rank;            // Store the rank r
    size_t num_parameters_dora;  // Total count of DORA parameters (A, B, m across layers)
    float* dora_temp_storage;    // Temporary storage for backward pass (reduced size)
    size_t dora_temp_storage_size;
} GPT2;

// Kaiming uniform initialization for LoRA A
void kaiming_uniform_init(float* tensor, size_t n, size_t fan_in) {
    float scale = sqrtf(3.0f / (float)fan_in);
    for(size_t i=0; i<n; ++i) {
        tensor[i] = ((float)rand() / RAND_MAX) * 2.0f * scale - scale;
    }
}

//Calculate L2 norm of a vector
float vector_norm(const float* vec, size_t size) {
    double sum_sq = 0.0;
#pragma omp parallel for reduction(+:sum_sq)
    for(size_t i = 0; i < size; ++i) {
        sum_sq += (double)vec[i] * (double)vec[i];
    }
    const double eps_norm_sq = 1e-10;
    return (float)sqrt(sum_sq + eps_norm_sq);
}

void* callocCheck(size_t num, size_t size) {
     void* ptr = calloc(num, size);
    if (ptr == NULL) {
        fprintf(stderr, "calloc failed\n");
        exit(EXIT_FAILURE);
    }
    return ptr;
}


void allocate_and_init_dora(GPT2 *model, size_t rank_r) {
    size_t L   = model->config.num_layers;
    size_t C   = model->config.channels;
    size_t NH  = model->config.num_heads;
    size_t NKVH = model->config.num_kv_heads;
    size_t HS  = C / NH;
    size_t q_dim = NH * HS;
    size_t kv_dim = NKVH * HS;
    size_t OC_qkv_gqa = q_dim + 2 * kv_dim;
    size_t OC_fc = 4 * C;
    size_t r = rank_r;
    model->dora_rank = r;
    int q_heads_per_kv_head = (NKVH == 0) ? 0 : NH / NKVH;

    printf("Initializing DORA (Rank %zu) and Adapting to GQA (NKVH %zu)...\n", r, NKVH);
    size_t mha_qkvw_layer_size = 3 * C * C;
    size_t mha_qkvb_layer_size = 3 * C;

    if (model->params.qkvw == NULL || model->params.qkvb == NULL) {
        fprintf(stderr, "ERROR in allocate_and_init_dora: Base qkvw or qkvb pointer is NULL before backup. Check loading.\n");
        exit(EXIT_FAILURE);
    }
     uintptr_t qkvb_addr = (uintptr_t)model->params.qkvb;
     if (qkvb_addr < 0x10000) {
         fprintf(stderr, "ERROR in allocate_and_init_dora: model->params.qkvb (%p) seems invalid before backup.\n", model->params.qkvb);
         exit(EXIT_FAILURE);
     }

    printf("Backing up original MHA weights and biases...\n");
    float* temp_mha_qkvw = (float*)mallocCheck(L * mha_qkvw_layer_size * sizeof(float));
    float* temp_mha_qkvb = (float*)mallocCheck(L * mha_qkvb_layer_size * sizeof(float));
    memcpy(temp_mha_qkvw, model->params.qkvw, L * mha_qkvw_layer_size * sizeof(float));
    memcpy(temp_mha_qkvb, model->params.qkvb, L * mha_qkvb_layer_size * sizeof(float));
    printf("Backup complete.\n");

    size_t size_qkv_A_layer = r * C;
    size_t size_qkv_B_layer = OC_qkv_gqa * r;
    size_t size_qkv_m_layer = OC_qkv_gqa;
    size_t size_att_A_layer = r * C;
    size_t size_att_B_layer = C * r;
    size_t size_att_m_layer = C;
    size_t size_fc_A_layer  = r * C;
    size_t size_fc_B_layer  = OC_fc * r;
    size_t size_fc_m_layer  = OC_fc;
    size_t size_fcp_A_layer = r * OC_fc;
    size_t size_fcp_B_layer = C * r;
    size_t size_fcp_m_layer = C;

    size_t total_dora_params = L * ( (size_qkv_A_layer + size_qkv_B_layer + size_qkv_m_layer) +
                                     (size_att_A_layer + size_att_B_layer + size_att_m_layer) +
                                     (size_fc_A_layer  + size_fc_B_layer  + size_fc_m_layer) +
                                     (size_fcp_A_layer + size_fcp_B_layer + size_fcp_m_layer) );
    model->num_parameters_dora = total_dora_params;
    printf("Allocating %.2f MB for DORA parameters (rank %zu, GQA NKVH %d)...\n",
           model->num_parameters_dora * sizeof(float) / (1024.0f * 1024.0f), r, (int)NKVH);

    model->params_memory_dora = (float*)mallocCheck(total_dora_params * sizeof(float));
    model->grads_memory_dora  = (float*)callocCheck(total_dora_params, sizeof(float));
    model->m_memory_dora      = (float*)callocCheck(total_dora_params, sizeof(float));
    model->v_memory_dora      = (float*)callocCheck(total_dora_params, sizeof(float));

    float* p_ptr = model->params_memory_dora;
    float* g_ptr = model->grads_memory_dora;
    // QKV
    model->params.qkvw_A = p_ptr; model->grads_dora.qkvw_A = g_ptr; p_ptr += L * size_qkv_A_layer; g_ptr += L * size_qkv_A_layer;
    model->params.qkvw_B = p_ptr; model->grads_dora.qkvw_B = g_ptr; p_ptr += L * size_qkv_B_layer; g_ptr += L * size_qkv_B_layer;
    model->params.qkvw_m = p_ptr; model->grads_dora.qkvw_m = g_ptr; p_ptr += L * size_qkv_m_layer; g_ptr += L * size_qkv_m_layer;
    // AttProj
    model->params.attprojw_A = p_ptr; model->grads_dora.attprojw_A = g_ptr; p_ptr += L * size_att_A_layer; g_ptr += L * size_att_A_layer;
    model->params.attprojw_B = p_ptr; model->grads_dora.attprojw_B = g_ptr; p_ptr += L * size_att_B_layer; g_ptr += L * size_att_B_layer;
    model->params.attprojw_m = p_ptr; model->grads_dora.attprojw_m = g_ptr; p_ptr += L * size_att_m_layer; g_ptr += L * size_att_m_layer;
    // FC
    model->params.fcw_A = p_ptr; model->grads_dora.fcw_A = g_ptr; p_ptr += L * size_fc_A_layer; g_ptr += L * size_fc_A_layer;
    model->params.fcw_B = p_ptr; model->grads_dora.fcw_B = g_ptr; p_ptr += L * size_fc_B_layer; g_ptr += L * size_fc_B_layer;
    model->params.fcw_m = p_ptr; model->grads_dora.fcw_m = g_ptr; p_ptr += L * size_fc_m_layer; g_ptr += L * size_fc_m_layer;
    // FCProj
    model->params.fcprojw_A = p_ptr; model->grads_dora.fcprojw_A = g_ptr; p_ptr += L * size_fcp_A_layer; g_ptr += L * size_fcp_A_layer;
    model->params.fcprojw_B = p_ptr; model->grads_dora.fcprojw_B = g_ptr; p_ptr += L * size_fcp_B_layer; g_ptr += L * size_fcp_B_layer;
    model->params.fcprojw_m = p_ptr; model->grads_dora.fcprojw_m = g_ptr;
    printf("DORA pointers assigned.\n");

    size_t size_qkv_bias_layer_gqa = OC_qkv_gqa;
    model->params.qkvb_gqa = (float*)mallocCheck(L * size_qkv_bias_layer_gqa * sizeof(float));
    if (model->params.qkvb_gqa == NULL) {
        fprintf(stderr, "ERROR: Failed to allocate memory for model->params.qkvb_gqa\n");
        exit(EXIT_FAILURE);
    }
    printf("Allocated memory for adapted GQA biases (qkvb_gqa) at address %p.\n", model->params.qkvb_gqa);

    printf("Starting MHA->GQA adaptation and DORA initialization loop...\n");
    size_t size_qkv_weights_layer_gqa = OC_qkv_gqa * C; // Adapted weight size

    for (int l = 0; l < L; ++l) {
        float* W0_qkv_mha_temp = temp_mha_qkvw + l * mha_qkvw_layer_size;
        float* b_qkv_mha_temp = temp_mha_qkvb + l * mha_qkvb_layer_size;
        float* W0_qkv_target = model->params.qkvw + l * size_qkv_weights_layer_gqa;
        float* b_qkv_target = model->params.qkvb_gqa + l * size_qkv_bias_layer_gqa;

        float* A_qkv = model->params.qkvw_A + l * size_qkv_A_layer;
        float* B_qkv = model->params.qkvw_B + l * size_qkv_B_layer;
        float* m_qkv = model->params.qkvw_m + l * size_qkv_m_layer;

        if (b_qkv_target == NULL || b_qkv_mha_temp == NULL) {
             fprintf(stderr, "ERROR Layer %d: b_qkv_target (%p) or b_qkv_mha_temp (%p) is NULL before memcpy.\n", l, b_qkv_target, b_qkv_mha_temp);
             exit(EXIT_FAILURE);
        }

        memcpy(W0_qkv_target, W0_qkv_mha_temp, q_dim * C * sizeof(float)); // Copy Q weights
        memcpy(b_qkv_target, b_qkv_mha_temp, q_dim * sizeof(float));     // Copy Q biases (CRASH WAS HERE)

        if (NKVH > 0 && q_heads_per_kv_head > 0) {
            #pragma omp parallel for schedule(static)
            for (int kv_group_idx = 0; kv_group_idx < NKVH; ++kv_group_idx) {
                int start_h = kv_group_idx * q_heads_per_kv_head;
                int end_h   = start_h + q_heads_per_kv_head;
                float* W0_k_target_group = W0_qkv_target + q_dim * C + kv_group_idx * HS * C;
                float* W0_v_target_group = W0_qkv_target + q_dim * C + kv_dim * C + kv_group_idx * HS * C;
                float* b_k_target_group = b_qkv_target + q_dim + kv_group_idx * HS;
                float* b_v_target_group = b_qkv_target + q_dim + kv_dim + kv_group_idx * HS;

                memset(W0_k_target_group, 0, HS * C * sizeof(float));
                memset(W0_v_target_group, 0, HS * C * sizeof(float));
                memset(b_k_target_group, 0, HS * sizeof(float));
                memset(b_v_target_group, 0, HS * sizeof(float));

                for (int orig_h = start_h; orig_h < end_h; ++orig_h) {
                    float* W0_k_mha_head = W0_qkv_mha_temp + C * C + orig_h * HS * C;
                    float* W0_v_mha_head = W0_qkv_mha_temp + 2 * C * C + orig_h * HS * C;
                    float* b_k_mha_head = b_qkv_mha_temp + C + orig_h * HS;
                    float* b_v_mha_head = b_qkv_mha_temp + 2 * C + orig_h * HS;
                    for (size_t i = 0; i < HS * C; ++i) {
                        W0_k_target_group[i] += W0_k_mha_head[i];
                        W0_v_target_group[i] += W0_v_mha_head[i];
                    }
                    for (size_t i = 0; i < HS; ++i) {
                        b_k_target_group[i] += b_k_mha_head[i];
                        b_v_target_group[i] += b_v_mha_head[i];
                    }
                }
                float inv_group_size = 1.0f / (float)q_heads_per_kv_head;
                for (size_t i = 0; i < HS * C; ++i) {
                    W0_k_target_group[i] *= inv_group_size;
                    W0_v_target_group[i] *= inv_group_size;
                }
                for (size_t i = 0; i < HS; ++i) {
                    b_k_target_group[i] *= inv_group_size;
                    b_v_target_group[i] *= inv_group_size;
                }
            }
        } else {
             fprintf(stderr, "Warning: Skipping GQA K/V adaptation in layer %d due to NKVH=%zu or NH/NKVH=%d\n", l, NKVH, q_heads_per_kv_head);
        }

        float eps_norm = 1e-5f;
        #pragma omp parallel for schedule(static)
        for (size_t o = 0; o < OC_qkv_gqa; ++o) {
            m_qkv[o] = vector_norm(W0_qkv_target + o * C, C);
             if (m_qkv[o] < eps_norm) { m_qkv[o] = eps_norm; }
        }
        kaiming_uniform_init(A_qkv, size_qkv_A_layer, C);
        memset(B_qkv, 0, size_qkv_B_layer * sizeof(float));

        float* W0_att = model->params.attprojw + l * C * C;
        float* A_att = model->params.attprojw_A + l * size_att_A_layer;
        float* B_att = model->params.attprojw_B + l * size_att_B_layer;
        float* m_att = model->params.attprojw_m + l * size_att_m_layer;
        #pragma omp parallel for schedule(static)
        for (size_t o = 0; o < C; ++o) {
            m_att[o] = vector_norm(W0_att + o * C, C);
            if (m_att[o] < eps_norm) { m_att[o] = eps_norm; }
        }
        kaiming_uniform_init(A_att, size_att_A_layer, C);
        memset(B_att, 0, size_att_B_layer * sizeof(float));

        float* W0_fc = model->params.fcw + l * OC_fc * C;
        float* A_fc = model->params.fcw_A + l * size_fc_A_layer;
        float* B_fc = model->params.fcw_B + l * size_fc_B_layer;
        float* m_fc = model->params.fcw_m + l * size_fc_m_layer;
        #pragma omp parallel for schedule(static)
        for (size_t o = 0; o < OC_fc; ++o) {
            m_fc[o] = vector_norm(W0_fc + o * C, C);
             if (m_fc[o] < eps_norm) { m_fc[o] = eps_norm; }
        }
        kaiming_uniform_init(A_fc, size_fc_A_layer, C);
        memset(B_fc, 0, size_fc_B_layer * sizeof(float));

        float* W0_fcp = model->params.fcprojw + l * C * OC_fc;
        float* A_fcp = model->params.fcprojw_A + l * size_fcp_A_layer;
        float* B_fcp = model->params.fcprojw_B + l * size_fcp_B_layer;
        float* m_fcp = model->params.fcprojw_m + l * size_fcp_m_layer;
        #pragma omp parallel for schedule(static)
        for (size_t o = 0; o < C; ++o) {
            m_fcp[o] = vector_norm(W0_fcp + o * OC_fc, OC_fc);
            if (m_fcp[o] < eps_norm) { m_fcp[o] = eps_norm; }
        }
        kaiming_uniform_init(A_fcp, size_fcp_A_layer, OC_fc);
        memset(B_fcp, 0, size_fcp_B_layer * sizeof(float));

    }
    printf("Adaptation and DORA initialization complete.\n");

    printf("Freeing temporary MHA buffers...\n");
    free(temp_mha_qkvw);
    free(temp_mha_qkvb);

    size_t max_OC_gqa_all = fmax(OC_qkv_gqa, fmax(C, OC_fc));
    size_t max_C_in_all = fmax(C, OC_fc);
    size_t max_r_C_in_all = fmax(r * C, r * OC_fc);
    size_t max_C_out_r_gqa_all = fmax(OC_qkv_gqa * r, fmax(C * r, OC_fc * r));
    size_t required_temp_elements = (max_OC_gqa_all * max_C_in_all) * 2 + max_r_C_in_all + max_C_out_r_gqa_all + max_OC_gqa_all;
    model->dora_temp_storage_size = required_temp_elements * sizeof(float);
    printf("Allocating %.2f MB for DORA backward temporary storage (estimated %zu elements)...\n",
           model->dora_temp_storage_size / (1024.0f * 1024.0f), required_temp_elements);
    model->dora_temp_storage = (float*)mallocCheck(model->dora_temp_storage_size);

    printf("\n--- Memory Usage Comparison (Post DORA/GQA Init) ---\n");
    size_t gqa_qkvw_count = L * size_qkv_weights_layer_gqa;
    size_t gqa_qkvb_count = L * size_qkv_bias_layer_gqa; // Using the new GQA bias count
    size_t base_attprojw_count = L * C * C;
    size_t base_attprojb_count = L * C;
    size_t base_fcw_count      = L * OC_fc * C;
    size_t base_fcb_count      = L * OC_fc;
    size_t base_fcprojw_count  = L * C * OC_fc;
    size_t base_fcprojb_count  = L * C;
    size_t other_base_params = model->num_parameters - L * (mha_qkvw_layer_size + mha_qkvb_layer_size + (C*C+C) + (OC_fc*C+OC_fc) + (C*OC_fc+C));
    if (other_base_params > model->num_parameters) other_base_params = 0;

    size_t total_base_params_after_adapt = gqa_qkvw_count + gqa_qkvb_count + base_attprojw_count + base_attprojb_count +
                                           base_fcw_count + base_fcb_count + base_fcprojw_count + base_fcprojb_count + other_base_params;
    size_t frozen_base_weights = gqa_qkvw_count + base_attprojw_count + base_fcw_count + base_fcprojw_count;
    size_t trainable_base_params_approx = total_base_params_after_adapt - frozen_base_weights;
    size_t dora_trainable_params_total = model->num_parameters_dora + trainable_base_params_approx;

    printf("Total Base Parameters (Original MHA Loaded): %zu (%.2f MB)\n", model->num_parameters,
           (model->num_parameters * sizeof(float)) / (1024.0f * 1024.0f));
     printf("Total Base Parameters (After GQA Adaptation): ~%zu (%.2f MB)\n", total_base_params_after_adapt,
           (total_base_params_after_adapt * sizeof(float)) / (1024.0f * 1024.0f));
    printf("Trainable DORA Parameters (A, B, m): %zu (%.2f MB)\n", model->num_parameters_dora,
           (model->num_parameters_dora * sizeof(float)) / (1024.0f * 1024.0f));
    printf("Trainable Base Parameters (Estimate: Non-Frozen Biases/Norms/Embed): ~%zu (%.2f MB)\n",
           trainable_base_params_approx, (trainable_base_params_approx * sizeof(float)) / (1024.0f * 1024.0f));
    printf("Total Trainable Params (DORA + Base Non-Frozen Estimate): ~%zu (%.2f MB)\n",
           dora_trainable_params_total, (dora_trainable_params_total * sizeof(float)) / (1024.0f * 1024.0f));
    double full_ft_memory_mb = (3.0 * model->num_parameters * sizeof(float)) / (1024.0 * 1024.0);
    double dora_ft_memory_refined_mb = ( (frozen_base_weights * sizeof(float)) +
                                         (trainable_base_params_approx * 3.0 * sizeof(float)) +
                                         (model->num_parameters_dora * 3.0 * sizeof(float))
                                       ) / (1024.0 * 1024.0);
    double memory_saved_mb = full_ft_memory_mb - dora_ft_memory_refined_mb;
    double savings_percent = (full_ft_memory_mb > 1e-6) ? (memory_saved_mb / full_ft_memory_mb) * 100.0 : 0.0;
    printf("Estimated Memory for Parameters + AdamW States:\n");
    printf("  - Full Fine-Tuning (Estimate based on MHA): %.2f MB\n", full_ft_memory_mb);
    printf("  - DORA Fine-Tuning (Refined Estimate): %.2f MB\n", dora_ft_memory_refined_mb);
    printf("Estimated Memory Saved with DORA: %.2f MB (%.1f%%)\n", memory_saved_mb, savings_percent);
    printf("---------------------------------\n\n");
}

void gpt2_build_from_checkpoint(GPT2 *model, const char* checkpoint_path) {

    // read in model from a checkpoint file
    FILE *model_file = fopenCheck(checkpoint_path, "rb");
    int model_header[256];
    freadCheck(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240326) { printf("Bad magic model file\n"); exit(1); }
    if (model_header[1] != 3) {
        printf("Bad version in model file\n");
        printf("---> HINT: try to re-run `python train_gpt2.py`\n");
        exit(1);
    }

    // read in hyperparameters
    size_t maxT, V, Vp, L, NH, C; // size_t to prevent int overflow
    model->config.max_seq_len = maxT = model_header[2];
    model->config.vocab_size = V = model_header[3];
    model->config.num_layers = L = model_header[4];
    model->config.num_heads = NH = model_header[5];
    model->config.channels = C = model_header[6];
    model->config.padded_vocab_size = Vp = model_header[7];
    printf("[GPT-2]\n");
    printf("max_seq_len: %zu\n", maxT);
    printf("vocab_size: %zu\n", V);
    printf("padded_vocab_size: %zu\n", Vp);
    printf("num_layers: %zu\n", L);
    printf("num_heads: %zu\n", NH);
    printf("channels: %zu\n", C);

    // allocate space for all the parameters and read them in
    fill_in_parameter_sizes(model->param_sizes,  model->config);

    // count the number of parameters
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += model->param_sizes[i];
    }
    printf("num_parameters: %zu\n", num_parameters);
    model->num_parameters = num_parameters;

    // read in all the parameters from file
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes);
    freadCheck(model->params_memory, sizeof(float), num_parameters, model_file);
    fcloseCheck(model_file);
    model->grads_memory_dora = NULL;
    model->m_memory_dora = NULL;
    model->v_memory_dora = NULL;
    model->acts_memory = NULL;
    model->grads_memory = NULL;
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->grads_acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f; // -1.0f will designate no loss
}

void dora_weight_sum(float* out, const float* base, const float* delta, size_t n) {
    for (size_t i = 0; i < n; i++) {
        out[i] = base[i] + delta[i];
    }
}


/*
 * This function simply executes the forward pass of GPT-2.
 *
 * Just takes a batch of input token sequences and processes them through multiple layers of teansofrmer network to predict the probability distribution
 * for the next token at each position in the sequences.
 *
 * During training, there will be target tokens and in that case a loss is calculated (cross-entropy).
 *
 */
void gpt2_forward(GPT2 *model, int* inputs, int* targets, size_t B, size_t T) {
    if (model->params_memory == NULL) {
        fprintf(stderr, "Error: Model parameters not initialized\n");
        exit(EXIT_FAILURE);
    }
    if (model->params.qkvw_A == NULL) { // Check DORA A params as proxy for DORA init
        fprintf(stderr, "Error: DORA parameters not initialized (A is NULL).\n");
        exit(EXIT_FAILURE);
    }


    // Model configuration parameters
    const size_t V = model->config.vocab_size; // vocubulary size
    const size_t Vp = model->config.padded_vocab_size;
    const size_t L = model->config.num_layers; //number of transformer layers
    const size_t NH = model->config.num_heads; // number of query heads in attention
    const size_t C = model->config.channels; //embedding dimension (model size)
    const size_t NKVH = model->config.num_kv_heads;  // Number of Key Value heads (GQA)
    const size_t HS = C / NH; //head size (dimension per head)
    const size_t q_dim = NH * HS; //query dimension
    const size_t kv_dim = NKVH * HS;
    const size_t qkv_out_dim = q_dim + 2 * kv_dim;  // GQA output dimension for fused QKV
    const size_t OC_fc = 4 * C;
    const size_t r = model->dora_rank;

    // Input validation
    #pragma omp parallel for
    for (int i = 0; i < B * T; i++) {
        if (inputs[i] < 0 || inputs[i] >= V) {
            fprintf(stderr, "E: input token %d @ %d\n", inputs[i], i);
            exit(1);
        }
        if (targets && (targets[i] < 0 || targets[i] >= V)) {
            fprintf(stderr, "E: target token %d @ %d\n", targets[i], i);
            exit(1);
        }
    }

    // Lazy allocation of activation memory (including ALL DORA caches)
    if (model->acts_memory == NULL) {
        printf("Lazily allocating activation memory...\n");
        model->batch_size = B;
        model->seq_len = T;
        //calculate required sizes for all the activation tensors
        fill_in_activation_sizes(model->act_sizes, model->config, B, T);
        size_t num_activations = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
            num_activations += model->act_sizes[i];
        }
        model->num_activations = num_activations;
        //Allocate one large block adn point struct members to the correct offsets
        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);
        //allocate memory to store copies of inputs and targets for the backward pass
        model->inputs = (int*)mallocCheck(B * T * sizeof(int));
        model->targets = (int*)mallocCheck(B * T * sizeof(int));
        printf("Activation memory allocated.\n");
    } else {
        if (B != model->batch_size || T != model->seq_len) {
            fprintf(stderr, "Error: Batch size/Seq len changed unexpectedly.\n");
            exit(1);
        }
    }

    // Cache inputs and targets
    memcpy(model->inputs, inputs, B * T * sizeof(int));
    if (targets) {
        memcpy(model->targets, targets, B * T * sizeof(int));
    }

    //set up pointers to model parameters and activations for ease of use later on
    ParameterTensors params = model->params;
    ActivationTensors acts = model->acts;
    //calculate intial embeddings by summing the token embeddings and position encoding
    encoder_forward(acts.encoded, inputs, params.wte, params.wpe, B, T, C);

    const float eps_norm = 1e-5f;

    // Process each transformer layer
    for (int l = 0; l < L; l++) {

        float* residual = (l == 0) ? acts.encoded : acts.residual3 + (l - 1) * B * T * C;

        // --- Layer Norm 1 ---
        //layer normalization before the attention block
        float* l_ln1 = acts.ln1 + l * B * T * C; //pointer to this layer's ln1 op
        layernorm_forward(l_ln1, acts.ln1_mean + l * B * T, acts.ln1_rstd + l * B * T,
                          residual, params.ln1w + l * C, params.ln1b + l * C, B, T, C);

        /*
         * DORA Based QKV projection to calculate attention
         */
        {
            size_t size_qkv_weights_layer = qkv_out_dim * C;
            //for DORA --> this is initial direction V --> that is frozen base weight matrix
            float* W0_adapted = params.qkvw + l * size_qkv_weights_layer;

            size_t size_qkv_bias_layer = qkv_out_dim;
            float* bias_adapted_gqa = params.qkvb_gqa + l * size_qkv_bias_layer;
            //here we are setting up the effective weight matrix for the QKV projection layer using DORA
            // directional update Delta_V is calculated using two trainable, low rank LORA matrices A and B.
            float* A = params.qkvw_A + l * r * C; // DORA A matrix
            float* B_lora = params.qkvw_B + l * qkv_out_dim * r; // DORA B matrix
            //qkvw_m is the trainable vector --> one value for each output dimension of the layer
            float* m = params.qkvw_m + l * qkv_out_dim;            // Dora Magnitude vector
            //pointers to the activation cahces for the dora intermediates
            // delta_V = B x A (lora compute)
            float* delta_cache = acts.qkv_delta_cache + l * qkv_out_dim * C; // Use GQA dimension
            // V' = V + delta_V = W0 (frozen weight = adapted GQA weight) + BA(lora)
            float* V_prime_cache = acts.qkv_V_prime_cache + l * qkv_out_dim * C; // Use GQA dimension
            //L2 normalized above
            float* norm_V_prime_cache = acts.qkv_norm_V_prime_cache + l * qkv_out_dim; // Use GQA dimension
            // point to the final effective dora weight matrix
            float* W_dora_cache = acts.qkv_W_dora_cache + l * qkv_out_dim * C; // Use GQA dimension

            //calculate LORA update --> delta = B * A
            matmul_forward(delta_cache, B_lora, A, NULL, qkv_out_dim, 1, r, C);

            //V' = frozen weights + delta (lora)
            #pragma omp parallel for
            for (size_t i = 0; i < qkv_out_dim * C; ++i) {
                V_prime_cache[i] = W0_adapted[i] + delta_cache[i]; // Use W0_adapted, not W0_gqa
            }

            //calculate column-wise norm
            #pragma omp parallel for
            for (size_t o = 0; o < qkv_out_dim; ++o) {
                norm_V_prime_cache[o] = vector_norm(V_prime_cache + o * C, C);
                 if (norm_V_prime_cache[o] < eps_norm) {
                     norm_V_prime_cache[o] = eps_norm;
                 }
            }

            // calculate final dora weight  W_dora = m * (V' / ||V'||c)
            //put back all that in the dora cache
            #pragma omp parallel for collapse(2)
            for (size_t o = 0; o < qkv_out_dim; ++o) {
                float inv_norm = 1.0f / (norm_V_prime_cache[o] /*+ eps_norm*/); // eps added implicitly by check above
                float scale = m[o] * inv_norm;
                for (size_t i = 0; i < C; ++i) {
                    W_dora_cache[o * C + i] = scale * V_prime_cache[o * C + i];
                }
            }

            //perform actual QKV projection qkv = ln1 * W_dora + bias
            matmul_forward(acts.qkv + l * B * T * qkv_out_dim, l_ln1, W_dora_cache, bias_adapted_gqa, B, T, C, qkv_out_dim);
        }

        // here is where we perform the attention --> with Q, K ,V projections
        float* l_atty = acts.atty + l * B * T * C;
        //--> takes in the projections --> and outputs attention, pre-softmqax scores
        attention_forward(l_atty,
                          acts.preatt + l * B * NH * T * T,
                          acts.att + l * B * NH * T * T,
                          acts.qkv + l * B * T * qkv_out_dim,
                          B, T, C, NH, model->config.num_kv_heads);

        // attention function above generates attention score for each token --> l_atty --> which basically
        // how much should each token attend to others
        // Now is attention output projection layer -->
        //project attention output back to model dimension C (input vector)--> using dora weights -->
        {
            float* W0 = params.attprojw + l * C * C;
            float* bias = params.attprojb + l * C;
            float* A = params.attprojw_A + l * r * C;
            float* B_lora = params.attprojw_B + l * C * r;
            float* m = params.attprojw_m + l * C;
            float* delta_cache = acts.attproj_delta_cache + l * C * C;
            float* V_prime_cache = acts.attproj_V_prime_cache + l * C * C;
            float* norm_V_prime_cache = acts.attproj_norm_V_prime_cache + l * C;
            float* W_dora_cache = acts.attproj_W_dora_cache + l * C * C;
            matmul_forward(delta_cache, B_lora, A, NULL, C, 1, r, C);
            #pragma omp parallel for
            for (size_t i = 0; i < C * C; ++i) V_prime_cache[i] = W0[i] + delta_cache[i];
            #pragma omp parallel for
            for (size_t o = 0; o < C; ++o) norm_V_prime_cache[o] = vector_norm(V_prime_cache + o * C, C);
            #pragma omp parallel for collapse(2)
            for (size_t o = 0; o < C; ++o) {
                 if (norm_V_prime_cache[o] < eps_norm) norm_V_prime_cache[o] = eps_norm; // Safety check
                float scale = m[o] / (norm_V_prime_cache[o] /*+ eps_norm*/);
                for (size_t i = 0; i < C; ++i) W_dora_cache[o * C + i] = scale * V_prime_cache[o * C + i];
            }
            //do the attention projection
            matmul_forward(acts.attproj + l * B * T * C, l_atty, W_dora_cache, bias, B, T, C, C);
        }

        // Residual connection 1
        // add the input of the residual layer to the output of the attention projection -->
        float* l_residual2 = acts.residual2 + l * B * T * C;
        residual_forward(l_residual2, residual, acts.attproj + l * B * T * C, B * T * C);


        // throw in another layer normalization
        float* l_ln2 = acts.ln2 + l * B * T * C;
        layernorm_forward(l_ln2, acts.ln2_mean + l * B * T, acts.ln2_rstd + l * B * T,
                          l_residual2, params.ln2w + l * C, params.ln2b + l * C, B, T, C);

        // MLP Feed forward layer -->
        // here we expand the dimension from C -> 4C --> once again we use DORA weights here
        {
            float* W0 = params.fcw + l * OC_fc * C;
            float* bias = params.fcb + l * OC_fc;
            float* A = params.fcw_A + l * r * C;
            float* B_lora = params.fcw_B + l * OC_fc * r;
            float* m = params.fcw_m + l * OC_fc;
            float* delta_cache = acts.fc_delta_cache + l * OC_fc * C;
            float* V_prime_cache = acts.fc_V_prime_cache + l * OC_fc * C;
            float* norm_V_prime_cache = acts.fc_norm_V_prime_cache + l * OC_fc;
            float* W_dora_cache = acts.fc_W_dora_cache + l * OC_fc * C;
            matmul_forward(delta_cache, B_lora, A, NULL, OC_fc, 1, r, C);
             #pragma omp parallel for
            for (size_t i = 0; i < OC_fc * C; ++i) V_prime_cache[i] = W0[i] + delta_cache[i];
            #pragma omp parallel for
            for (size_t o = 0; o < OC_fc; ++o) norm_V_prime_cache[o] = vector_norm(V_prime_cache + o * C, C);
            #pragma omp parallel for collapse(2)
            for (size_t o = 0; o < OC_fc; ++o) {
                 if (norm_V_prime_cache[o] < eps_norm) norm_V_prime_cache[o] = eps_norm; // Safety check
                float scale = m[o] / (norm_V_prime_cache[o] /*+ eps_norm*/);
                for (size_t i = 0; i < C; ++i) W_dora_cache[o * C + i] = scale * V_prime_cache[o * C + i];
            }
            matmul_forward(acts.fch + l * B * T * OC_fc, l_ln2, W_dora_cache, bias, B, T, C, OC_fc);
        }

        //elementwise GELU
        float* l_fch_gelu = acts.fch_gelu + l * B * T * OC_fc;
        gelu_forward(l_fch_gelu, acts.fch + l * B * T * OC_fc, B * T * OC_fc);

        //MLP-2 --> reduce dimension from 4*C ---> C
        {
             float* W0 = params.fcprojw + l * C * OC_fc;
             float* bias = params.fcprojb + l * C;
             float* A = params.fcprojw_A + l * r * OC_fc;
             float* B_lora = params.fcprojw_B + l * C * r;
             float* m = params.fcprojw_m + l * C;
             float* delta_cache = acts.fcproj_delta_cache + l * C * OC_fc;
             float* V_prime_cache = acts.fcproj_V_prime_cache + l * C * OC_fc;
             float* norm_V_prime_cache = acts.fcproj_norm_V_prime_cache + l * C;
             float* W_dora_cache = acts.fcproj_W_dora_cache + l * C * OC_fc;
             matmul_forward(delta_cache, B_lora, A, NULL, C, 1, r, OC_fc);
              #pragma omp parallel for
             for (size_t i = 0; i < C * OC_fc; ++i) V_prime_cache[i] = W0[i] + delta_cache[i];
             #pragma omp parallel for
             for (size_t o = 0; o < C; ++o) norm_V_prime_cache[o] = vector_norm(V_prime_cache + o * OC_fc, OC_fc);
             #pragma omp parallel for collapse(2)
             for (size_t o = 0; o < C; ++o) {
                 if (norm_V_prime_cache[o] < eps_norm) norm_V_prime_cache[o] = eps_norm; // Safety check
                 float scale = m[o] / (norm_V_prime_cache[o] /*+ eps_norm*/);
                 for (size_t i = 0; i < OC_fc; ++i) W_dora_cache[o * OC_fc + i] = scale * V_prime_cache[o * OC_fc + i];
             }
             matmul_forward(acts.fcproj + l * B * T * C, l_fch_gelu, W_dora_cache, bias, B, T, OC_fc, C);
        }

        // fcproj = fchgelu * W_dora + bias
        residual_forward(acts.residual3 + l * B * T * C,
                         l_residual2,
                         acts.fcproj + l * B * T * C,
                         B * T * C);
    }

    // Final layer normalization -->
    float* final_residual = acts.residual3 + (L - 1) * B * T * C;
    layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd,
                      final_residual, params.lnfw, params.lnfb,
                      B, T, C);

    // Logits calculation
    matmul_forward(acts.logits, acts.lnf, params.wte, NULL, B, T, C, Vp);
    //softmax
    softmax_forward(acts.probs, acts.logits, B, T, V, Vp);

    // Lsos calculation
    if (targets) {
        crossentropy_forward(model->acts.losses, acts.probs, targets, B, T, Vp);
        double loss_sum = 0.0;
        #pragma omp parallel for reduction(+:loss_sum)
        for (size_t i = 0; i < B * T; i++) {
            loss_sum += model->acts.losses[i];
        }
        model->mean_loss = (float)(loss_sum / (B * T));
    } else {
        model->mean_loss = -1.0f;
    }
}

// this is just to recent the gradients from the previous pass -->

void gpt2_zero_grad(GPT2 *model) {
    if(model->grads_memory != NULL) {
        memset(model->grads_memory, 0, model->num_parameters * sizeof(float));
    }
    if(model->grads_acts_memory != NULL) {
        memset(model->grads_acts_memory, 0, model->num_activations * sizeof(float));
    }

    if(model->grads_memory_dora != NULL) {
        memset(model->grads_memory_dora, 0, model->num_parameters_dora * sizeof(float));

        assert(model->grads_dora.qkvw_A != NULL);
        assert(model->grads_dora.attprojw_A != NULL);
        assert(model->grads_dora.fcw_A != NULL);
        assert(model->grads_dora.fcprojw_A != NULL);
    }
}

void matmul_transpose_B(float* out, const float* A, const float* B, int m, int n, int k) {
#pragma omp parallel for collapse(2)
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int i = 0; i < k; ++i) {
                sum += A[row * k + i] * B[col * k + i];
            }
            out[row * n + col] = sum;
        }
    }
}


void matmul_transpose_A(float* out, const float* A, const float* B, int m, int n, int k) {

#pragma omp parallel for collapse(2)
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;

            for (int i = 0; i < k; ++i) {
                sum += A[i * m + row] * B[i * n + col];
            }
            out[row * n + col] = sum;
        }
    }
}

void gpt2_backward(GPT2 *model) {
    if (model->mean_loss == -1.0f) {
        printf("Error: must forward with targets before backward\n");
        exit(1);
    }

    // if this is the first backward pass --> allocate memory to store the gradients
    if (model->grads_memory == NULL) {
        printf("Lazily allocating gradient memory...\n");

        model->grads_memory = malloc_and_point_parameters(&model->grads, model->param_sizes);
        model->grads_acts_memory = malloc_and_point_activations(&model->grads_acts, model->act_sizes);

        size_t L = model->config.num_layers;
        size_t C = model->config.channels;
        size_t NH = model->config.num_heads;
        size_t NKVH = model->config.num_kv_heads;
        size_t HS = C / NH;
        size_t q_dim = NH * HS;
        size_t kv_dim = NKVH * HS;
        size_t OC_qkv_gqa = q_dim + 2 * kv_dim;
        size_t OC_fc = 4 * C;
        size_t r = model->dora_rank;

        size_t size_qkv_A_layer = r * C;
        size_t size_qkv_B_layer = OC_qkv_gqa * r;
        size_t size_qkv_m_layer = OC_qkv_gqa;
        size_t size_att_A_layer = r * C;
        size_t size_att_B_layer = C * r;
        size_t size_att_m_layer = C;
        size_t size_fc_A_layer  = r * C;
        size_t size_fc_B_layer  = OC_fc * r;
        size_t size_fc_m_layer  = OC_fc;
        size_t size_fcp_A_layer = r * OC_fc;
        size_t size_fcp_B_layer = C * r;
        size_t size_fcp_m_layer = C;

        size_t total_dora_grads_size =
            L * ((size_qkv_A_layer + size_qkv_B_layer + size_qkv_m_layer) +
                 (size_att_A_layer + size_att_B_layer + size_att_m_layer) +
                 (size_fc_A_layer  + size_fc_B_layer  + size_fc_m_layer) +
                 (size_fcp_A_layer + size_fcp_B_layer + size_fcp_m_layer));
        //allocate one big chunk of memory and have pointers point to different sections for different gradients
        model->grads_memory_dora = (float*)calloc(total_dora_grads_size, sizeof(float));
        if (model->grads_memory_dora == NULL) { exit(1); }
        printf("Allocated %.2f MB for ALL DORA gradients\n",
               total_dora_grads_size * sizeof(float) / (1024.0f*1024.0f));

        // Point the pointers within the grads_dora structure.
        float* g_ptr = model->grads_memory_dora;
        model->grads_dora.qkvw_A = g_ptr; g_ptr += L * size_qkv_A_layer;
        model->grads_dora.qkvw_B = g_ptr; g_ptr += L * size_qkv_B_layer;
        model->grads_dora.qkvw_m = g_ptr; g_ptr += L * size_qkv_m_layer;
        model->grads_dora.attprojw_A = g_ptr; g_ptr += L * size_att_A_layer;
        model->grads_dora.attprojw_B = g_ptr; g_ptr += L * size_att_B_layer;
        model->grads_dora.attprojw_m = g_ptr; g_ptr += L * size_att_m_layer;
        model->grads_dora.fcw_A = g_ptr; g_ptr += L * size_fc_A_layer;
        model->grads_dora.fcw_B = g_ptr; g_ptr += L * size_fc_B_layer;
        model->grads_dora.fcw_m = g_ptr; g_ptr += L * size_fc_m_layer;
        model->grads_dora.fcprojw_A = g_ptr; g_ptr += L * size_fcp_A_layer;
        model->grads_dora.fcprojw_B = g_ptr; g_ptr += L * size_fcp_B_layer;
        model->grads_dora.fcprojw_m = g_ptr;
        gpt2_zero_grad(model);
        printf("Gradient memory allocated and zeroed.\n");
    }
    if (model->grads_memory == NULL) {
         printf("ERROR: Base gradient memory is NULL after allocation block.\n");
         exit(1);
    }
    if (model->num_parameters_dora > 0 && model->grads_dora.qkvw_A == NULL) {
         printf("ERROR: DORA gradient pointer (qkvw_A) is NULL after allocation block.\n");
         exit(1);
    }

    size_t B = model->batch_size;
    size_t T = model->seq_len;
    size_t V = model->config.vocab_size;
    size_t Vp = model->config.padded_vocab_size;
    size_t L = model->config.num_layers;
    size_t NH = model->config.num_heads;
    size_t NKVH = model->config.num_kv_heads; // for GQA
    size_t C = model->config.channels;
    size_t HS = C / NH;
    size_t q_dim = NH * HS;
    size_t kv_dim = NKVH * HS;
    size_t qkv_out_dim = q_dim + 2 * kv_dim;
    size_t OC_fc = 4 * C;
    size_t r = model->dora_rank;

    /*
     * Temporary storage for DORA backward pass
     * will be reused across DORA layers
     *
     */
    size_t max_OC = fmax(qkv_out_dim, fmax(C, OC_fc));
    size_t max_C_in = fmax(C, OC_fc);
    size_t max_r_C_in = fmax(r * C, r * OC_fc);
    size_t max_C_out_r = fmax(qkv_out_dim * r, fmax(C * r, OC_fc * r));
    float* grad_dL_dWdora = model->dora_temp_storage;          // Size: max_OC * max_C_in
    float* grad_V_prime   = grad_dL_dWdora + max_OC * max_C_in;  // Size: max_OC * max_C_in
    float* temp_grad_A    = grad_V_prime + max_OC * max_C_in;      // Size: max_r_C_in
    float* temp_grad_B    = temp_grad_A + max_r_C_in;              // Size: max_C_out_r

    //initialize the gradients for backpropagation
    float dloss_mean = 1.0f / (B * T);
    #pragma omp parallel for
    for (size_t i = 0; i < B * T; i++) {
        model->grads_acts.losses[i] = dloss_mean;
    }

    //backpropagate from behind -->

    //backprop through cross-entropy and softmax together
    crossentropy_softmax_backward(model->grads_acts.logits, model->grads_acts.losses,
                                  model->acts.probs, model->targets, B, T, V, Vp);
    //backprop through the final logits calculation

    matmul_backward(model->grads_acts.lnf, model->grads.wte, NULL,
                    model->grads_acts.logits, model->acts.lnf, model->params.wte,
                    B, T, C, Vp);

    //backprop through final layer normalization
    float* final_residual = model->acts.residual3 + (L - 1) * B * T * C;
    float* dresidual_final = model->grads_acts.residual3 + (L - 1) * B * T * C;
    layernorm_backward(dresidual_final, model->grads.lnfw, model->grads.lnfb,
                       model->grads_acts.lnf, final_residual, model->params.lnfw,
                       model->acts.lnf_mean, model->acts.lnf_rstd, B, T, C);

    //backprop through transformer layers
    for (int l = L - 1; l >= 0; l--) {
        float* residual = (l == 0) ? model->acts.encoded : model->acts.residual3 + (l - 1) * B * T * C;
        float* dresidual = (l == 0) ? model->grads_acts.encoded : model->grads_acts.residual3 + (l - 1) * B * T * C;
        float* dresidual_current_layer = model->grads_acts.residual3 + l * B * T * C;

        float* dl_ln1w = model->grads.ln1w + l * C;
        float* dl_ln1b = model->grads.ln1b + l * C;
        float* dl_attprojb = model->grads.attprojb + l * C;
        float* dl_ln2w = model->grads.ln2w + l * C;
        float* dl_ln2b = model->grads.ln2b + l * C;
        float* dl_fcb = model->grads.fcb + l * OC_fc;
        float* dl_fcprojb = model->grads.fcprojb + l * C;
        float* dl_qkvb = model->grads.qkvb + l * qkv_out_dim; // Using new dimension

        float* l_ln1 = model->acts.ln1 + l * B * T * C;
        float* l_ln1_mean = model->acts.ln1_mean + l * B * T;
        float* l_ln1_rstd = model->acts.ln1_rstd + l * B * T;
        float* l_atty = model->acts.atty + l * B * T * C;
        float* l_att = model->acts.att + l * B * NH * T * T;
        float* l_residual2 = model->acts.residual2 + l * B * T * C;
        float* l_ln2 = model->acts.ln2 + l * B * T * C;
        float* l_ln2_mean = model->acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = model->acts.ln2_rstd + l * B * T;
        float* l_fch = model->acts.fch + l * B * T * OC_fc;
        float* l_fch_gelu = model->acts.fch_gelu + l * B * T * OC_fc;
        float* l_qkv = model->acts.qkv + l * B * T * qkv_out_dim; // New GQA size
        float* dl_atty = model->grads_acts.atty + l * B * T * C;
        float* dl_preatt = model->grads_acts.preatt + l * B * NH * T * T;
        float* dl_att = model->grads_acts.att + l * B * NH * T * T;
        float* dl_attproj = model->grads_acts.attproj + l * B * T * C;
        float* dl_residual2 = model->grads_acts.residual2 + l * B * T * C;
        float* dl_ln2 = model->grads_acts.ln2 + l * B * T * C;
        float* dl_fch = model->grads_acts.fch + l * B * T * OC_fc;
        float* dl_fch_gelu = model->grads_acts.fch_gelu + l * B * T * OC_fc;
        float* dl_fcproj = model->grads_acts.fcproj + l * B * T * C;
        float* dl_ln1_target = model->grads_acts.ln1 + l * B * T * C;
        float* dl_qkv = model->grads_acts.qkv + l * B * T * qkv_out_dim; // New GQA size

        residual_backward(dl_residual2, dl_fcproj, dresidual_current_layer, B * T * C);

        {
            const size_t FCProj_InpDim = OC_fc; // Input to FCProj is MLP hidden dim
            const size_t FCProj_OutDim = C;     // Output of FCProj is C
            float* grad_A = model->grads_dora.fcprojw_A + l * r * FCProj_InpDim; // Grad for A (r x OC_fc)
            float* grad_B = model->grads_dora.fcprojw_B + l * FCProj_OutDim * r; // Grad for B (C x r)
            float* grad_m = model->grads_dora.fcprojw_m + l * FCProj_OutDim;     // Grad for m (C)
            float* l_A = model->params.fcprojw_A + l * r * FCProj_InpDim;    // Param A (r x OC_fc)
            float* l_B = model->params.fcprojw_B + l * FCProj_OutDim * r;    // Param B (C x r)
            float* l_m = model->params.fcprojw_m + l * FCProj_OutDim;        // Param m (C)
            float* cached_V_prime = model->acts.fcproj_V_prime_cache + l * FCProj_OutDim * FCProj_InpDim;
            float* cached_norm_V_prime = model->acts.fcproj_norm_V_prime_cache + l * FCProj_OutDim;
            float* cached_W_dora = model->acts.fcproj_W_dora_cache + l * FCProj_OutDim * FCProj_InpDim;
            float* dl_fcprojb_target = model->grads.fcprojb + l * FCProj_OutDim;
            float* l_fch_gelu_input = model->acts.fch_gelu + l * B * T * FCProj_InpDim; // Input activation


            memset(grad_dL_dWdora, 0, FCProj_OutDim * FCProj_InpDim * sizeof(float)); // Use correct size
            matmul_backward(NULL, grad_dL_dWdora, dl_fcprojb_target, // Pass correct bias grad pointer
                            dl_fcproj, l_fch_gelu_input, NULL,        // Pass correct input activation
                            B, T, FCProj_InpDim, FCProj_OutDim);      // Use correct dimensions

            const float eps_div = 1e-5f;

            #pragma omp parallel for
            for (size_t o = 0; o < FCProj_OutDim; ++o) {
                float norm_o = cached_norm_V_prime[o];
                float norm_o_plus_eps = norm_o + eps_div;
                float norm_o_inv_robust = 1.0f / norm_o_plus_eps;
                float m_o = l_m[o];
                float* grad_W_dora_row = grad_dL_dWdora + o * FCProj_InpDim;
                float* V_prime_row = cached_V_prime + o * FCProj_InpDim;
                float* grad_V_prime_row = grad_V_prime + o * FCProj_InpDim;

                double dot_product = 0.0;
                for (size_t i = 0; i < FCProj_InpDim; ++i) {
                    dot_product += (double)grad_W_dora_row[i] * (double)V_prime_row[i];
                }
                float grad_m_o = (float)(dot_product * norm_o_inv_robust);
                #pragma omp atomic update
                grad_m[o] += grad_m_o;

                float scale_m_norm_robust = m_o * norm_o_inv_robust;
                float term2_scale = (float)dot_product / (norm_o_plus_eps * norm_o_plus_eps);
                for (size_t i = 0; i < FCProj_InpDim; ++i) {
                     grad_V_prime_row[i] = scale_m_norm_robust * (grad_W_dora_row[i] - term2_scale * V_prime_row[i]);
                }
            }

            memset(temp_grad_A, 0, r * FCProj_InpDim * sizeof(float));
            memset(temp_grad_B, 0, FCProj_OutDim * r * sizeof(float));

            matmul_transpose_A(temp_grad_A, l_B, grad_V_prime, r, FCProj_InpDim, FCProj_OutDim);

            matmul_transpose_B(temp_grad_B, grad_V_prime, l_A, FCProj_OutDim, r, FCProj_InpDim);

            #pragma omp parallel for
            for (size_t i = 0; i < r * FCProj_InpDim; ++i) grad_A[i] += temp_grad_A[i]; // Accumulate grad A
            #pragma omp parallel for
            for (size_t i = 0; i < FCProj_OutDim * r; ++i) grad_B[i] += temp_grad_B[i]; // Accumulate grad B


            matmul_backward(dl_fch_gelu, NULL, NULL,
                            dl_fcproj, NULL, cached_W_dora,
                            B, T, FCProj_InpDim, FCProj_OutDim);
        }

        gelu_backward(dl_fch, l_fch, dl_fch_gelu, B * T * OC_fc);

        {
            float* grad_A = model->grads_dora.fcw_A + l * r * C;
            float* grad_B = model->grads_dora.fcw_B + l * OC_fc * r;
            float* grad_m = model->grads_dora.fcw_m + l * OC_fc;
            float* l_A = model->params.fcw_A + l * r * C;
            float* l_B = model->params.fcw_B + l * OC_fc * r;
            float* l_m = model->params.fcw_m + l * OC_fc;
            float* cached_V_prime = model->acts.fc_V_prime_cache + l * OC_fc * C;
            float* cached_norm_V_prime = model->acts.fc_norm_V_prime_cache + l * OC_fc;
            float* cached_W_dora = model->acts.fc_W_dora_cache + l * OC_fc * C;
            memset(grad_dL_dWdora, 0, OC_fc * C * sizeof(float));
            matmul_backward(NULL, grad_dL_dWdora, dl_fcb, dl_fch, l_ln2, NULL, B, T, C, OC_fc);
            float eps_norm = 1e-5f;
            memset(temp_grad_A, 0, r * C * sizeof(float));
            memset(temp_grad_B, 0, OC_fc * r * sizeof(float));
            matmul_transpose_A(temp_grad_A, l_B, grad_V_prime, r, C, OC_fc);
            matmul_transpose_B(temp_grad_B, grad_V_prime, l_A, OC_fc, r, C);
            #pragma omp parallel for
            for (size_t i = 0; i < r * C; ++i) grad_A[i] += temp_grad_A[i];
            #pragma omp parallel for
            for (size_t i = 0; i < OC_fc * r; ++i) grad_B[i] += temp_grad_B[i];
            matmul_backward(dl_ln2, NULL, NULL, dl_fch, NULL, cached_W_dora, B, T, C, OC_fc);
        }

        layernorm_backward(dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2,
                           model->params.ln2w + l * C, l_ln2_mean, l_ln2_rstd, B, T, C);
        residual_backward(dresidual, dl_attproj, dl_residual2, B * T * C);

        {
            float* grad_A = model->grads_dora.attprojw_A + l * r * C;
            float* grad_B = model->grads_dora.attprojw_B + l * C * r;
            float* grad_m = model->grads_dora.attprojw_m + l * C;
            float* l_A = model->params.attprojw_A + l * r * C;
            float* l_B = model->params.attprojw_B + l * C * r;
            float* l_m = model->params.attprojw_m + l * C;
            float* cached_V_prime = model->acts.attproj_V_prime_cache + l * C * C;
            float* cached_norm_V_prime = model->acts.attproj_norm_V_prime_cache + l * C;
            float* cached_W_dora = model->acts.attproj_W_dora_cache + l * C * C;
            memset(grad_dL_dWdora, 0, C * C * sizeof(float));
            matmul_backward(NULL, grad_dL_dWdora, dl_attprojb, dl_attproj, l_atty, NULL, B, T, C, C);
            float eps_norm = 1e-5f;
            memset(temp_grad_A, 0, r * C * sizeof(float));
            memset(temp_grad_B, 0, C * r * sizeof(float));
            matmul_transpose_A(temp_grad_A, l_B, grad_V_prime, r, C, C);
            matmul_transpose_B(temp_grad_B, grad_V_prime, l_A, C, r, C);
            #pragma omp parallel for
            for (size_t i = 0; i < r * C; ++i) grad_A[i] += temp_grad_A[i];
            #pragma omp parallel for
            for (size_t i = 0; i < C * r; ++i) grad_B[i] += temp_grad_B[i];
            matmul_backward(dl_atty, NULL, NULL, dl_attproj, NULL, cached_W_dora, B, T, C, C);
        }

        attention_backward(dl_qkv, dl_preatt, dl_att, dl_atty,
                           l_qkv, l_att,
                           B, T, C, NH, model->config.num_kv_heads);

        {
            float* grad_A   = model->grads_dora.qkvw_A + l * r * C;
            float* grad_B   = model->grads_dora.qkvw_B + l * qkv_out_dim * r; // GQA dim
            float* grad_m   = model->grads_dora.qkvw_m + l * qkv_out_dim;     // GQA dim
            float* l_A      = model->params.qkvw_A + l * r * C;
            float* l_B      = model->params.qkvw_B + l * qkv_out_dim * r;     // GQA dim
            float* l_m      = model->params.qkvw_m + l * qkv_out_dim;         // GQA dim
            float* cached_V_prime = model->acts.qkv_V_prime_cache + l * qkv_out_dim * C; // GQA dim
            float* cached_norm_V_prime = model->acts.qkv_norm_V_prime_cache + l * qkv_out_dim; // GQA dim
            float* cached_W_dora = model->acts.qkv_W_dora_cache + l * qkv_out_dim * C; // GQA dim

            const size_t InpDim = C;        // Input dimension is C.
            const size_t OutDim = qkv_out_dim; // Output dimension is the new GQA dimension.

            memset(grad_dL_dWdora, 0, OutDim * InpDim * sizeof(float));
            matmul_backward(NULL, grad_dL_dWdora, dl_qkvb, dl_qkv, l_ln1, NULL, B, T, InpDim, OutDim);

            const float eps_div = 1e-5f;

            #pragma omp parallel for // Parallelize over output dimension 'o'
            for (size_t o = 0; o < OutDim; ++o) {

                float norm_o = cached_norm_V_prime[o];
                float norm_o_plus_eps = norm_o + eps_div;
                float norm_o_inv_robust = 1.0f / norm_o_plus_eps;

                float m_o = l_m[o];
                float* grad_W_dora_row = grad_dL_dWdora + o * InpDim;
                float* V_prime_row = cached_V_prime + o * InpDim;
                float* grad_V_prime_row = grad_V_prime + o * InpDim;

                double dot_product = 0.0;
                for (size_t i = 0; i < InpDim; ++i) {
                    dot_product += (double)grad_W_dora_row[i] * (double)V_prime_row[i];
                }

                float grad_m_o = (float)(dot_product * norm_o_inv_robust);
                #pragma omp atomic update
                grad_m[o] += grad_m_o;
                float scale_m_norm_robust = m_o * norm_o_inv_robust;
                float term2_scale = (float)dot_product / (norm_o_plus_eps * norm_o_plus_eps);

                for (size_t i = 0; i < InpDim; ++i) {
                     grad_V_prime_row[i] = scale_m_norm_robust * (grad_W_dora_row[i] - term2_scale * V_prime_row[i]);
                }
            }

            memset(temp_grad_A, 0, r * InpDim * sizeof(float));
            memset(temp_grad_B, 0, OutDim * r * sizeof(float));

            matmul_transpose_A(temp_grad_A, l_B, grad_V_prime, r, InpDim, OutDim);

            matmul_transpose_B(temp_grad_B, grad_V_prime, l_A, OutDim, r, InpDim);

            #pragma omp parallel for
            for (size_t i = 0; i < r * InpDim; ++i)
                grad_A[i] += temp_grad_A[i];
            #pragma omp parallel for
            for (size_t i = 0; i < OutDim * r; ++i)
                grad_B[i] += temp_grad_B[i];

            matmul_backward(dl_ln1_target, NULL, NULL, dl_qkv, NULL, cached_W_dora, B, T, InpDim, OutDim);

        }

        layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_ln1_target,
                           residual, model->params.ln1w + l * C, l_ln1_mean, l_ln1_rstd,
                           B, T, C);
    }

    encoder_backward(model->grads.wte, model->grads.wpe,
                      model->grads_acts.encoded, model->inputs, B, T, C);
}

//basically this is the AdamW optimizer
// at this point the gpt2_backward has already calculated the gradients
// now this bad boy will apply those optimizr changes to modify the models tranable parameters
void gpt2_update(GPT2 *model, float learning_rate, float beta1, float beta2,
                 float eps, float weight_decay, int t) {
    // lazy allocation of optimizer states
    //Adamw needs to keep track of momentum and variance for each parameter
    if (model->m_memory == NULL) {
        model->m_memory = (float*)calloc(model->num_parameters, sizeof(float));
        model->v_memory = (float*)calloc(model->num_parameters, sizeof(float));
        assert(model->m_memory != NULL && model->v_memory != NULL);
        printf("Warning: Base optimizer states allocated in gpt2_update.\n");
    }

    //for dora parameters
    if (model->m_memory_dora == NULL && model->num_parameters_dora > 0) {
        model->m_memory_dora = (float*)calloc(model->num_parameters_dora, sizeof(float));
        model->v_memory_dora = (float*)calloc(model->num_parameters_dora, sizeof(float));
        assert(model->m_memory_dora != NULL && model->v_memory_dora != NULL);
        printf("Warning: DORA optimizer states allocated in gpt2_update.\n");
    }

    if (model->num_parameters_dora == 0) {
        fprintf(stderr, "Warning: Trying to update DORA params, but none seem allocated.\n");
    }

    size_t L = model->config.num_layers;
    size_t C = model->config.channels;
    size_t NH = model->config.num_heads;
    size_t NKVH = model->config.num_kv_heads;
    size_t HS = C / NH;
    size_t q_dim = NH * HS;           // Dimension for Q projections
    size_t kv_dim = NKVH * HS;        // Dimension for K and V projections (GQA)
    size_t OC_qkv_gqa = q_dim + 2 * kv_dim;
    size_t OC_fc  = 4 * C;           // Hidden dimension for FFN.
    size_t r = model->dora_rank;

    size_t size_qkv_A_layer = r * C;         // A maps from C
    size_t size_qkv_B_layer = OC_qkv_gqa * r;
    size_t size_qkv_m_layer = OC_qkv_gqa;

    size_t size_att_A_layer = r * C;
    size_t size_att_B_layer = C * r;
    size_t size_att_m_layer = C;

    size_t size_fc_A_layer  = r * C;
    size_t size_fc_B_layer  = OC_fc * r;
    size_t size_fc_m_layer  = OC_fc;

    size_t size_fcp_A_layer = r * OC_fc;
    size_t size_fcp_B_layer = C * r;
    size_t size_fcp_m_layer = C;


    //offsets with the flat DORA arrays
    size_t current_dora_offset = 0;

    //QKV offsets
    size_t dora_offset_qkv_A = current_dora_offset;
    current_dora_offset += L * size_qkv_A_layer;
    size_t dora_offset_qkv_B = current_dora_offset;
    current_dora_offset += L * size_qkv_B_layer;
    size_t dora_offset_qkv_m = current_dora_offset;
    current_dora_offset += L * size_qkv_m_layer;

    //AttProj offsets
    size_t dora_offset_att_A = current_dora_offset;
    current_dora_offset += L * size_att_A_layer;
    size_t dora_offset_att_B = current_dora_offset;
    current_dora_offset += L * size_att_B_layer;
    size_t dora_offset_att_m = current_dora_offset;
    current_dora_offset += L * size_att_m_layer;
    // FC Offsets.
    size_t dora_offset_fc_A = current_dora_offset;
    current_dora_offset += L * size_fc_A_layer;
    size_t dora_offset_fc_B = current_dora_offset;
    current_dora_offset += L * size_fc_B_layer;
    size_t dora_offset_fc_m = current_dora_offset;
    current_dora_offset += L * size_fc_m_layer;
    // FCProj Offsets.
    size_t dora_offset_fcp_A = current_dora_offset;
    current_dora_offset += L * size_fcp_A_layer;
    size_t dora_offset_fcp_B = current_dora_offset;
    current_dora_offset += L * size_fcp_B_layer;
    size_t dora_offset_fcp_m = current_dora_offset;


    float beta1_correction = 1.0f - powf(beta1, t);
    float beta2_correction = 1.0f - powf(beta2, t);

    size_t current_base_offset = 0;
    for (int j = 0; j < NUM_PARAMETER_TENSORS; j++) {
        size_t num_param_tensor = model->param_sizes[j];

        //best part of this entire project
        // skip all frozen base weights
        // 4-> qkvw, 6-> attprojw, 10-> fcw, 12-> fcprojw
        if (j == 4 || j == 6 || j == 10 || j == 12) {

            current_base_offset += num_param_tensor;
            continue;
        }


        float* param_ptr = model->params_memory + current_base_offset;
        float* grad_ptr  = model->grads_memory + current_base_offset;
        float* m_ptr     = model->m_memory + current_base_offset;
        float* v_ptr     = model->v_memory + current_base_offset;

        float current_weight_decay = weight_decay;
        //determine weight decay
        if (j == 0 // wte
            || j == 1 // wpe
            || j == 2 || j == 3 //ln1w,ln1b
            || j == 5 || j == 7 || // qkvb, attprojb
            j == 8 || j == 9 // ln2w, ln2b
            || j == 11 || j == 13 //fcb, fcprojb
            || j == 14 || j == 15) { // lnfw, lnfb
            current_weight_decay = 0.0f;
        }

        #pragma omp parallel for
        for (size_t i = 0; i < num_param_tensor; i++) {
            float param = param_ptr[i];
            float grad  = grad_ptr[i];
            float m_val = beta1 * m_ptr[i] + (1.0f - beta1) * grad;
            float v_val = beta2 * v_ptr[i] + (1.0f - beta2) * grad * grad;
            m_ptr[i] = m_val;
            v_ptr[i] = v_val;
            float m_hat = m_val / beta1_correction;
            float v_hat = v_val / beta2_correction;
            param_ptr[i] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + current_weight_decay * param);
        }
        current_base_offset += num_param_tensor;
    }

    if (model->num_parameters_dora > 0) {
        #pragma omp parallel for
        for (int l = 0; l < L; ++l) {
            {
                // Parameter pointers
                float* pA = model->params.qkvw_A + l * size_qkv_A_layer;
                float* pB = model->params.qkvw_B + l * size_qkv_B_layer; // GQA size
                float* pm = model->params.qkvw_m + l * size_qkv_m_layer; // GQA size
                // Gradient pointers
                float* gA = model->grads_dora.qkvw_A + l * size_qkv_A_layer;
                float* gB = model->grads_dora.qkvw_B + l * size_qkv_B_layer; // GQA size
                float* gm = model->grads_dora.qkvw_m + l * size_qkv_m_layer; // GQA size
                // Optimizer state pointers (using calculated GQA offsets)
                float* mA = model->m_memory_dora + dora_offset_qkv_A + l * size_qkv_A_layer;
                float* vA = model->v_memory_dora + dora_offset_qkv_A + l * size_qkv_A_layer;
                float* mB = model->m_memory_dora + dora_offset_qkv_B + l * size_qkv_B_layer; // GQA offset/size
                float* vB = model->v_memory_dora + dora_offset_qkv_B + l * size_qkv_B_layer; // GQA offset/size
                float* mm = model->m_memory_dora + dora_offset_qkv_m + l * size_qkv_m_layer; // GQA offset/size
                float* vm = model->v_memory_dora + dora_offset_qkv_m + l * size_qkv_m_layer; // GQA offset/size

                // Update qkvw_A (size: r * C)
                for (size_t i = 0; i < size_qkv_A_layer; ++i) {
                    float param = pA[i]; float grad = gA[i];
                    float m_val = beta1 * mA[i] + (1.f - beta1) * grad;
                    float v_val = beta2 * vA[i] + (1.f - beta2) * grad * grad;
                    mA[i] = m_val; vA[i] = v_val;
                    float m_hat = m_val / beta1_correction; float v_hat = v_val / beta2_correction;
                    pA[i] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param); // Apply WD to A/B
                }
                // Update qkvw_B (size: OC_qkv_gqa * r)
                for (size_t i = 0; i < size_qkv_B_layer; ++i) {
                    float param = pB[i]; float grad = gB[i];
                    float m_val = beta1 * mB[i] + (1.f - beta1) * grad;
                    float v_val = beta2 * vB[i] + (1.f - beta2) * grad * grad;
                    mB[i] = m_val; vB[i] = v_val;
                    float m_hat = m_val / beta1_correction; float v_hat = v_val / beta2_correction;
                    pB[i] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param); // Apply WD to A/B
                }
                // Update qkvw_m (size: OC_qkv_gqa) - NO Weight Decay on magnitude
                for (size_t i = 0; i < size_qkv_m_layer; ++i) {
                    float grad = gm[i]; // param = pm[i]; (not needed for WD=0)
                    float m_val = beta1 * mm[i] + (1.f - beta1) * grad;
                    float v_val = beta2 * vm[i] + (1.f - beta2) * grad * grad;
                    mm[i] = m_val; vm[i] = v_val;
                    float m_hat = m_val / beta1_correction; float v_hat = v_val / beta2_correction;
                    pm[i] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps)); // No WD term
                }
            }

            {
                float* pA = model->params.attprojw_A + l * size_att_A_layer;
                float* gA = model->grads_dora.attprojw_A + l * size_att_A_layer;
                float* pB = model->params.attprojw_B + l * size_att_B_layer;
                float* gB = model->grads_dora.attprojw_B + l * size_att_B_layer;
                float* pm = model->params.attprojw_m + l * size_att_m_layer;
                float* gm = model->grads_dora.attprojw_m + l * size_att_m_layer;
                float* mA = model->m_memory_dora + dora_offset_att_A + l * size_att_A_layer;
                float* vA = model->v_memory_dora + dora_offset_att_A + l * size_att_A_layer;
                float* mB = model->m_memory_dora + dora_offset_att_B + l * size_att_B_layer;
                float* vB = model->v_memory_dora + dora_offset_att_B + l * size_att_B_layer;
                float* mm = model->m_memory_dora + dora_offset_att_m + l * size_att_m_layer;
                float* vm = model->v_memory_dora + dora_offset_att_m + l * size_att_m_layer;
                for (size_t i = 0; i < size_att_A_layer; ++i) { /* AdamW update for A */ }
                for (size_t i = 0; i < size_att_B_layer; ++i) { /* AdamW update for B */ }
                for (size_t i = 0; i < size_att_m_layer; ++i) { /* AdamW update for m (no WD) */ }
            }

            {
                 float* pA = model->params.fcw_A + l * size_fc_A_layer;
                 float* gA = model->grads_dora.fcw_A + l * size_fc_A_layer;
                 float* pB = model->params.fcw_B + l * size_fc_B_layer;
                 float* gB = model->grads_dora.fcw_B + l * size_fc_B_layer;
                 float* pm = model->params.fcw_m + l * size_fc_m_layer;
                 float* gm = model->grads_dora.fcw_m + l * size_fc_m_layer;
                 float* mA = model->m_memory_dora + dora_offset_fc_A + l * size_fc_A_layer;
                 float* vA = model->v_memory_dora + dora_offset_fc_A + l * size_fc_A_layer;
                 float* mB = model->m_memory_dora + dora_offset_fc_B + l * size_fc_B_layer;
                 float* vB = model->v_memory_dora + dora_offset_fc_B + l * size_fc_B_layer;
                 float* mm = model->m_memory_dora + dora_offset_fc_m + l * size_fc_m_layer;
                 float* vm = model->v_memory_dora + dora_offset_fc_m + l * size_fc_m_layer;
                for (size_t i = 0; i < size_fc_A_layer; ++i) { /* AdamW update for A */ }
                for (size_t i = 0; i < size_fc_B_layer; ++i) { /* AdamW update for B */ }
                for (size_t i = 0; i < size_fc_m_layer; ++i) { /* AdamW update for m (no WD) */ }
            }

            {
                 float* pA = model->params.fcprojw_A + l * size_fcp_A_layer;
                 float* gA = model->grads_dora.fcprojw_A + l * size_fcp_A_layer;
                 float* pB = model->params.fcprojw_B + l * size_fcp_B_layer;
                 float* gB = model->grads_dora.fcprojw_B + l * size_fcp_B_layer;
                 float* pm = model->params.fcprojw_m + l * size_fcp_m_layer;
                 float* gm = model->grads_dora.fcprojw_m + l * size_fcp_m_layer;
                 float* mA = model->m_memory_dora + dora_offset_fcp_A + l * size_fcp_A_layer;
                 float* vA = model->v_memory_dora + dora_offset_fcp_A + l * size_fcp_A_layer;
                 float* mB = model->m_memory_dora + dora_offset_fcp_B + l * size_fcp_B_layer;
                 float* vB = model->v_memory_dora + dora_offset_fcp_B + l * size_fcp_B_layer;
                 float* mm = model->m_memory_dora + dora_offset_fcp_m + l * size_fcp_m_layer;
                 float* vm = model->v_memory_dora + dora_offset_fcp_m + l * size_fcp_m_layer;
                for (size_t i = 0; i < size_fcp_A_layer; ++i) { /* AdamW update for A */ }
                for (size_t i = 0; i < size_fcp_B_layer; ++i) { /* AdamW update for B */ }
                for (size_t i = 0; i < size_fcp_m_layer; ++i) { /* AdamW update for m (no WD) */ }
            }
        }

    }

}

void gpt2_free(GPT2 *model) {

    free(model->params_memory);
    if(model->grads_memory != NULL) { free(model->grads_memory); }
    if(model->m_memory != NULL) { free(model->m_memory); }
    if(model->v_memory != NULL) { free(model->v_memory); }
    if(model->acts_memory != NULL) { free(model->acts_memory); }
    if(model->grads_acts_memory != NULL) { free(model->grads_acts_memory); }
    if(model->inputs != NULL) { free(model->inputs); }
    if(model->targets != NULL) { free(model->targets); }

    if(model->params_memory_dora != NULL) { free(model->params_memory_dora); }
    if(model->grads_memory_dora != NULL) { free(model->grads_memory_dora); }
    if(model->m_memory_dora != NULL) { free(model->m_memory_dora); }
    if(model->v_memory_dora != NULL) { free(model->v_memory_dora); }
    if(model->dora_temp_storage != NULL) { free(model->dora_temp_storage); }
}

#ifndef TESTING
// if we are TESTING (see test_gpt2.c), we'll skip the int main below
// ----------------------------------------------------------------------------
// sampler

unsigned int random_u32(uint64_t *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(uint64_t *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

// ----------------------------------------------------------------------------
// main training loop
int main() {

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
    allocate_and_init_dora(&model, 8);
    // build the DataLoaders from tokens files. for now use tiny_shakespeare if available, else tiny_stories
    const char* tiny_stories_train = "dev/data/tinystories/TinyStories_train.bin";
    const char* tiny_stories_val = "dev/data/tinystories/TinyStories_val.bin";
    const char* tiny_shakespeare_train = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    const char* tiny_shakespeare_val = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
    const char* train_tokens = access(tiny_shakespeare_train, F_OK) != -1 ? tiny_shakespeare_train : tiny_stories_train;
    const char* val_tokens = access(tiny_shakespeare_val, F_OK) != -1 ? tiny_shakespeare_val : tiny_stories_val;
    int B = 4; // batch size 4 (i.e. 4 independent token sequences will be trained on)
    int T = 64; // sequence length 64 (i.e. each sequence is 64 tokens long). must be <= maxT, which is 1024 for GPT-2
    DataLoader train_loader, val_loader;
    dataloader_init(&train_loader, train_tokens, B, T, 0, 1, 1);
    dataloader_init(&val_loader, val_tokens, B, T, 0, 1, 0);
    printf("train dataset num_batches: %zu\n", train_loader.num_tokens / (B*T));
    printf("val dataset num_batches: %zu\n", val_loader.num_tokens / (B*T));
    int val_num_batches = 5;

    // build the Tokenizer
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    // some memory for generating samples from the model
    uint64_t rng_state = 1337;
    int* gen_tokens = (int*)mallocCheck(B * T * sizeof(int));
    const int genT = 64; // number of steps of inference we will do

    // train
    struct timespec start, end;
    for (int step = 0; step <= 100; step++) {

        // once in a while estimate the validation loss
        if (step % 10 == 0) {
            float val_loss = 0.0f;
            dataloader_reset(&val_loader);
            for (int i = 0; i < val_num_batches; i++) {
                dataloader_next_batch(&val_loader);
                gpt2_forward(&model, val_loader.inputs, val_loader.targets, B, T);
                val_loss += model.mean_loss;
            }
            val_loss /= val_num_batches;
            printf("val loss %f\n", val_loss);
        }

        // once in a while do model inference to print generated text
        if (step > 0 && step % 20 == 0) {
            // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
            for(int i = 0; i < B * T; ++i) {
                gen_tokens[i] = tokenizer.eot_token;
            }
            // now sample from the model autoregressively
            printf("generating:\n---\n");
            for (int t = 1; t < genT; t++) {
                // note that inference is very wasteful here because for each token
                // we re-calculate the forward pass for all of (B,T) positions from scratch
                // but the inference here is just for sanity checking anyway
                // and we can maybe optimize a bit more later, with careful tests
                gpt2_forward(&model, gen_tokens, NULL, B, T);
                // furthermore, below we're only using b=0 (i.e. the first row) of all B rows
                // we're in principle running B "inference streams" in parallel here
                // but only using position 0
                // get the Vp-dimensional vector probs[0, t-1, :]
                float* probs = model.acts.probs + (t-1) * model.config.padded_vocab_size;
                float coin = random_f32(&rng_state);
                // note we're only sampling from the first V elements, ignoring padding
                // (the probabilities in the padded region should be zero anyway)
                int next_token = sample_mult(probs, model.config.vocab_size, coin);
                gen_tokens[t] = next_token;
                // print the generated token, either using the Tokenizer or a fallback
                if (tokenizer.init_ok) {
                    const char* token_str = tokenizer_decode(&tokenizer, next_token);
                    safe_printf(token_str);
                } else {
                    // fall back to printing the token id
                    printf("%d ", next_token);
                }
                fflush(stdout);
            }
            printf("\n---\n");
        }

        // do a training step
        clock_gettime(CLOCK_MONOTONIC, &start);
        dataloader_next_batch(&train_loader);
        gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);
        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, step+1);
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("step %d: train loss %f (took %f ms)\n", step, model.mean_loss, time_elapsed_s * 1000);
    }

    // free
    dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    tokenizer_free(&tokenizer);
    gpt2_free(&model);
    free(gen_tokens);
    return 0;
}
#endif
