/* Jonas B. Nilsson 2025
Add option to include bi-lstm*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"

static double squash( a ) double a; { return (1.0 / (1.0 + exp( -a ))); }
int debug=0;

int p_seed;
WORD    p_alphabet;
int p_niter;
FILENAME    p_testfile;
int p_verbose;
int p_dumptype;
float   p_blnorm;
int     p_bl;
FILENAME        p_blosummat;
int     p_bls;
int p_freq;
FILENAME    p_blrealmat;
float   p_min;
float   p_max;
float   p_eta;
float   p_bplim;
int p_printpred;
int p_initweight;
int p_useRELU_initweight;
float   p_weight;
WORD p_nhid;
int p_n_lstm;
int p_hs_dim;
FILENAME        p_syn;
int p_teststop;
int p_ntest;
int p_strict;
int     p_xrescale;
int     p_encmhc;
FILENAME p_mhclist;
int     p_ninput;
int     p_allowX;
int     p_burnin;
FILENAME    p_filterlist;
int p_adam;
float p_etaLSTM;
float   p_beta1;
float   p_beta2;
float   p_epsilon;
float   p_dropout;
int p_sampleweight;
FILENAME p_reftcrfile;
float       p_lambda;
float   p_dropout_lstm;
int p_bilstm;


PARAM   param[] = {
    "-s", VINT      p_seed, "Seed [-1] Default, [0] Time [>0] Specific seed", "-1",
    "-a", VWORD     p_alphabet, "Amino acid alphabet", "ARNDCQEGHILKMFPSTWYV",
    "-i", VINT      p_niter, "Number of minization cycles", "300",
    "-ft", VFNAME   p_testfile, "File with test data", "",
    "-v", VSWITCH   p_verbose, "Verbose mode", "0",
    "-dtype", VINT  p_dumptype, "Test performance type [0] Square Error [1] Pearsons CC", "0",
    "-mpat", VFNAME p_blosummat, "Blosum Matrix pattern", "/Users/mniel/data/matrices/BLOSUM%i",
        "-bln", VFLOAT p_blnorm, "Normalizing factor for blosum score", "5.0",
        "-bl", VSWITCH p_bl, "Use Blosum encoding [default is sparse]", "0",
        "-bls",  VINT  p_bls, "BLOSUM SCORE MATRIX",       "50",
    "-freq", VSWITCH p_freq, "Use amino acid frequency for PFR encoding [default is Blosum]", "0",
    "-blf", VFNAME p_blrealmat, "Real value blosum matrix filename", "/Users/mniel/data/matrices/blosum62.freq_rownorm",
    "-max", VFLOAT  p_max, "Max value in sparse encoding", "0.9",
        "-min", VFLOAT  p_min, "Min value in sparse encoding", "0.05",
    "-eta", VFLOAT  p_eta, "Eta for weight update", "0.05",
    "-etaLSTM", VFLOAT  p_etaLSTM, "Eta for weight update in LSTM", "0.005",
    "-blim", VFLOAT   p_bplim, "Limit for backpropagation", "0.00001",
    "-ppred", VSWITCH p_printpred, "Print predictions", "0",
    "-iw", VSWITCH  p_initweight, "Init weights randomly [defaults is as HOWLIN]", "0",
    "-iwReLU", VSWITCH p_useRELU_initweight, "Use ReLU weight initialization", "0",
        "-w",  VFLOAT   p_weight, "Initial value for weight", "0.1",
    "-nh", VWORD p_nhid, "Number of hidden neurons (comma-separated list)", "1",
    "-nlstm", VINT p_n_lstm, "Number of LSTMs per input feature", "1",
    "-hsdim", VINT p_hs_dim, "Dimension of LSTM hidden state", "8",
    "-syn", VFNAME  p_syn,  "Name of synaps file", "syn.dat",
    "-teststop", VSWITCH p_teststop, "Stop training on best test performance", "0",
    "-nt", VINT     p_ntest, "Test interval", "10",
    "-strict", VSWITCH p_strict, "Strict mode", "0",
    "-xrsc", VSWITCH p_xrescale, "Exclude rescale synapse to binary encoding", "0",
    "-aX", VSWITCH p_allowX, "Allow X in sequences", "0",
    "-dropout", VFLOAT p_dropout, "Dropout rate", "-1",
    "-dropout_lstm", VFLOAT p_dropout_lstm, "Dropout rate for the LSTM input", "-1",
    "-sw", VSWITCH p_sampleweight, "Use sample weighting for loss (include weights as 3rd column in tcrfile)", "0",
    "-l", VFLOAT    p_lambda, "Weight on square uncert", "0.05",
    "-adam", VSWITCH p_adam, "Use Adam in CNN", "0",
    "-beta1", VFLOAT p_beta1, "Beta1 value for Adam", "0.9",
    "-beta2", VFLOAT p_beta2, "Beta2 value for Adam", "0.999",
    "-epsilon", VFLOAT p_epsilon, "Epsilon value for Adam", "0.00000001",
    "-bilstm", VSWITCH p_bilstm, "Use bi-lstm (default is forward lstm)", "0",
    "-burnin_trash", VINT     p_burnin, "Burn-in before trash assignment", "-1",
    0
};


void setnet(float **x, SYNAPS *syn)

{
    int l;
    for (l = 1; l < syn->nlay; l++)
        x[l][syn->nnlay[l]] = 1.0;
}

void tcrlist2encode_chain(TCRLIST *tcrlist, int alen, float **blmat)

{

    TCRLIST *tl;
    CHAINLIST *cl;
    int i, j, k, n, m;
    int ix;
    int l;

    for (j = 0, tl = tcrlist; tl; tl = tl->next, j++) {

        for (n=0; n < tl->n_inp; n++) {
            cl = tl->chains[n];

            cl->enc = fmatrix(0, cl->len - 1, 0, alen - 1);
            //cl->i = ivector(0, cl->len - 1);

            for (i = 0; i < cl->len; i++) {

                ix = strpos(p_alphabet, cl->seq[i]);

                if (p_strict && ix < 0) {
                    printf("Error. Unknown letter %c %s\n", cl->seq[i], p_alphabet);
                    exit(1);
                }
                //cl->i[i] = ix; // save amino acid indices

                for (k = 0; k < alen; k++) {
                    if (p_bl && ix >= 0)
                        cl->enc[i][k] = blmat[ix][k];
                    else
                        cl->enc[i][k] = (ix == k ? p_max : p_min);
                }
            }

            cl->oenc = cl->enc; // oenc holds the forward encoding

            if (p_bilstm) {

                cl->enc = fmatrix(0, cl->len - 1, 0, alen - 1);

                for (m=0, i = cl->len-1; i >= 0; i--, m++) {

                    ix = strpos(p_alphabet, cl->seq[i]);

                    if (p_strict && ix < 0) {
                        printf("Error. Unknown letter %c %s\n", cl->seq[i], p_alphabet);
                        exit(1);
                    }
                    //cl->i[i] = ix; // save amino acid indices

                    for (k = 0; k < alen; k++) {
                        if (p_bl && ix >= 0)
                            cl->enc[m][k] = blmat[ix][k];
                        else
                            cl->enc[m][k] = (ix == k ? p_max : p_min);
                    }
                }
                cl->ienc = cl->enc; // ienc holds the reverse encoding
            }
        }
    }
}


void forward_inp(float **x, SYNAPS *s, float *inp, int alen, int train)

{
    int i, l, k;
    float a;
    float **wgt_pl;
    float *x_pl, *x_l;
    float *wgt_pl_pk;

    setnet(x, s);

    for (i = 0; i < s->nnlay[1]; i++) {

        x[1][i] = inp[i]; // concatenated output from LSTM layers
        //printf("%f\n",inp[i]);

        if (p_dropout > -1 && train) {

            if (drand48() < p_dropout)
                x[1][i] = 0.0;
            else
                x[1][i] /= (1.0-p_dropout);
        }

        if (debug)
            printf("Test %i %f\n", i, x[1][i]);
    }
    //exit(1);

    for (l = 2; l < s->nlay; l++) {

        x_pl = x[l - 1];
        x_l = x[l];
        wgt_pl = s->wgt[l - 1];

        for (k = 0; k < s->nnlay[l]; k++) {

            wgt_pl_pk = wgt_pl[k];

            a = 0.0;

            for (i = 0; i < s->nnlay[l - 1] + 1; i++) //include bias weight
                a += x_pl[i] * wgt_pl_pk[i];

            x_l[k] = squash(a);
        }
    }
}


void forward_lstm(TCRLIST *tl, SYNAPS *s, float *inp, int n_inp, int inp_dim,
                float ***Z, float ***F, float ***I, float ***C, float ***O, float ***S, float ***H, float *dropout_mask, int *inv_vector)

{
    int i,l,t,n, h,z, j, m;
    CHAINLIST *chain;
    int total_inp = inp_dim + p_hs_dim + 1;
    float *wgt_mh;
    float z_val;
    float f_val, i_val, c_val, o_val;

    /* Each LSTM has four gates, each with a weight matrix and a bias vector

    Forget_gate (F)      w_F : ( hidden_size * input_size ), b_F : (hidden_size * 1)
    Input_gate (I)       w_I : ( hidden_size * input_size ), b_I : (hidden_size * 1)
    Candidate_gate (C)   w_C : ( hidden_size * input_size ), b_C : (hidden_size * 1)
    Output_gate (O)      w_O : ( hidden_size * input_size ), b_O : (hidden_size * 1)

    The input to the LSTM at timepoint t is Z, which is the sequence encoding input at position t contatenated with the t-1 hidden state

    The four gates are computed as follows

    F[t] = sigmoid( w_F · Z[t] + b_F )
    I[t] = sigmoid(W_i · Z[t] + b_i)
    C[t] = tanh(W_c · Z[t] + b_c)
    O[t] = sigmoid(W_o · Z[t] + b_o)

    where · is matrix,vector product

    Furthermore, we have to compute the cell state S and hidden state H

    S[t] = F[t]*S[t-1] + I[t]*C[t]
    H[t] = O[t] * tanh(S[t])

    where * and + are elementwise multiplication and addition, respectively
    */


    // loop over input chains
    for (i=0; i < n_inp; i++ ) {

        chain = tl->chains[i];

        // loop over LSTMs
        for (l=0; l < p_n_lstm ; l++) { // if bi-lstm, p_n_lstm = p_n_lstm*2

            // pre-compute indexes
            j = i*p_n_lstm+l; // counter for LSTMs
            m = j * p_hs_dim; // counter for the first dimension of FICO weight matrices

            // Before we forward this datapoint, we make sure to reset LSTM memory
            // by setting the values stored in each gate, cell state and hidden state to zero
            for (t=0; t < chain->len; t++) {
                for (h=0; h < p_hs_dim; h++) {
                    F[j][t][h] = 0.0; // Forget gate
                    I[j][t][h] = 0.0; // Input gate
                    C[j][t][h] = 0.0; // Candidate gate
                    O[j][t][h] = 0.0; // Output gate
                    S[j][t][h] = 0.0; // Cell state
                    H[j][t][h] = 0.0; // Hidden state
                }
            }

            // use same dropout mask per time step

            if (dropout_mask) {
          
                for (n = 0; n < inp_dim + p_hs_dim; n++) {
                    if (drand48() < p_dropout_lstm)
                        dropout_mask[n] = 0.0;
                    else
                        dropout_mask[n] = 1.0 / (1.0 - p_dropout_lstm); // inverted dropout scaling
                }
            }

            if (inv_vector[j] == 1)
                chain->enc = chain->ienc;
            else
                chain->enc = chain->oenc;

            // loop over timesteps - i.e. sequence positions
            for (t=0; t < chain->len; t++) {

                // copy input and previous hidden state to Z vector
                for (n=0; n < inp_dim; n++) {
                    Z[j][t][n] = chain->enc[t][n];

                    if (dropout_mask)
                        Z[j][t][n] *= dropout_mask[n];
                }

                if (t > 0) {
                    for (h = 0; h < p_hs_dim; h++) {
                        Z[j][t][inp_dim + h] = H[j][t-1][h];

                        if (dropout_mask)
                            Z[j][t][inp_dim + h] *= dropout_mask[h];
                    }
                }
                else
                    for (h=0; h < p_hs_dim; h++)
                        Z[j][t][inp_dim + h] = 0.0;

                // set bias input
                Z[j][t][inp_dim + p_hs_dim] = 1.0;

                // The next two loops perform the weight matrix multiplication
                // loop over hidden dimension
                for (h=0; h < p_hs_dim; h++) {
                    // loop over Z input length
                    for (z=0; z < total_inp; z++) { // total_inp = inp_dim + p_hs_dim + 1 (+1 to include bias)

                        wgt_mh = s->wgt[0][m + h];

                        z_val = Z[j][t][z];
                        //printf("z %f\n", z_val);
                        // each wgt[0][m] holds four weight matrices (F I C O)
                        F[j][t][h] += wgt_mh[0 * total_inp + z]*z_val;
                        I[j][t][h] += wgt_mh[1 * total_inp + z]*z_val;
                        C[j][t][h] += wgt_mh[2 * total_inp + z]*z_val;
                        O[j][t][h] += wgt_mh[3 * total_inp + z]*z_val;
                    }
                }

                // apply activation functions and calculate cell state and hidden state
                for (h=0; h < p_hs_dim; h++) {

                    F[j][t][h] = squash(F[j][t][h]);
                    I[j][t][h] = squash(I[j][t][h]);
                    C[j][t][h] = tanh(C[j][t][h]);
                    O[j][t][h] = squash(O[j][t][h]);

                    f_val = F[j][t][h];
                    i_val = I[j][t][h];
                    c_val = C[j][t][h];
                    o_val = O[j][t][h];

                    if (t > 0) // Previous cell state is only defined for t > 0
                        S[j][t][h] = f_val * S[j][t-1][h] + i_val * c_val;
                    else
                        S[j][t][h] = i_val * c_val;

                    H[j][t][h] = o_val * tanh(S[j][t][h]);

                }
            }

            // After the last time step, we copy the hidden state vector to the FFNN input layer
            for (h=0; h < p_hs_dim; h++) {
                //printf("%f\n",H[j][chain->len-1][h]);
                inp[j * p_hs_dim + h] = H[j][chain->len-1][h];
            }
        }
    }

    if (debug)
        printf("Done forward_lstm\n");

}

void backprob(TCRLIST *tl, float **x, SYNAPS *s, float target, float sw, int n_inp, int inp_dim,
    float ***Z, float ***F, float ***I, float ***C, float ***O, float ***S, float ***H,
    float **moments, float **velocity, float b1p, float b2p, float beta1_min, float beta2_min)
{
    int j, k, l, i, h, n, z, t, m;
    float ***wgt;
    float delta3, d_o;
    float gprime_o, gprime_o0, gprime_o1, gprime, gprime_h1k, g;
    float deltaH;
    float S_act;
    int outlay;
    CHAINLIST *chain;
    int total_inp;
    float *wgt_mh;
    float z_val;
    float f_val, i_val, c_val, o_val;
    float mom_hat, vel_hat;

    float **delta = fmatrix(0, s->nlay-1, 0, s->maxnper);

    wgt = s->wgt; // 3d matrix: layer index, number of neurons in next layer, number of neurons in current layer

    outlay = s->nlay - 1; // output layer index

    // Backpropagation in output layer
    d_o = (x[outlay][0] - target);

    if (p_sampleweight)
        d_o *= sw;

    gprime_o0 = x[outlay][0] * ( 1 - x[outlay][0] );
    gprime_o1 = x[outlay][1] * ( 1 - x[outlay][1] );

    delta[outlay][0] = x[outlay][1] * d_o * gprime_o0;
    delta[outlay][1] = ( 0.5 * d_o * d_o - 2 * p_lambda * ( 1 - x[outlay][1]) ) * gprime_o1;

    for (j = 0; j <= s->nnlay[outlay - 1]; j++) { // includes the bias
        wgt[outlay - 1][0][j] -= p_eta * delta[outlay][0] * x[outlay - 1][j]; // pred
        wgt[outlay - 1][1][j] -= p_eta * delta[outlay][1] * x[outlay - 1][j]; // relia
    }

    // Hidden layers
    for (l = outlay - 1; l > 1; l--) {  // Start at first hidden layer

        // Loop over neurons in current hidden layer
        for (j = 0; j < s->nnlay[l]; j++) {

            delta[l][j] = 0.0;  // Initialize delta for this neuron

            for (i = 0; i < s->nnlay[l + 1]; i++) // Loop over neurons in next layer
                delta[l][j] += delta[l+1][i] * wgt[l][i][j];

            // Calculate gprime_h2j and update weights for the previous layer
            gprime = x[l][j] * (1 - x[l][j]);
            delta[l][j] *= gprime;

            // Loop over neurons in previous layer
            for (k = 0; k <= s->nnlay[l - 1]; k++) { //includes the bias
                wgt[l - 1][j][k] -= p_eta * delta[l][j] * x[l - 1][k];
            }
        }
    }

    // Backpropagation through time in LSTM Layers

    // https://medium.com/@dovanhuyen2018/back-propagation-in-long-short-term-memory-lstm-a13ad8ae7a57
    // https://courses.engr.illinois.edu/ece417/fa2020/slides/lec19.pdf

    // F[t] = sigmoid(X[t]·w_f + H[t-1]·v_f + b_f) // Forget
    // I[t] = sigmoid(X[t]·w_i + H[t-1]·v_i + b_i) // Input
    // C[t] = tanh(X[t]·w_c + H[t-1]·v_c + b_c)    // Candidate
    // O[t] = sigmoid(X[t]·w_o + H[t-1]·v_o + b_o) // Output
    // S[t] = F[t]*S[t-1] + I[t]*C[t]              // State (cell state)
    // H[t] = O[t]*tanh(S[t])                      // Hidden (hidden state)

    // where w are the weights connected to the sequence input, and v are weights connected to previous hidden state vectors

    // We need the derivatives of E with respect to w_f, v_f, w_i, v_i, w_c, v_c, w_o, v_o

    // First, using the chain rule we get
    // dE/dF[t] = dE/dS[t] * dS[t]/dF[t] = dE/dS[t] * S[t-1]
    // dE/dI[t] = dE/dS[t] * dS[t]/dI[t] = dE/dS[t] * C[t]
    // dE/dC[t] = dE/dS[t] * dS[t]/dC[t] = dE/dS[t] * I[t]
    // dE/dO[t] = dE/dH[t] * dH[t]/dO[t] = dE/dH[t] * tanh(S[t])

    // Next we define these quantities

    // F[t] = sigmoid(A_f[t]) , A_f[t] = X[t]·w_f + H[t-1]·v_f + b_f
    // I[t] = sigmoid(A_i[t]) , A_i[t] = X[t]·w_i + H[t-1]·v_i + b_i
    // C[t] = tanh(A_c[t])    , A_c[t] = X[t]·w_c + H[t-1]·v_c + b_c
    // O[t] = sigmoid(A_o[t]) , A_o[t] = X[t]·w_o + H[t-1]·v_o + b_o

    // dE/dA_f[t] = dE/dF[t] * dF[t]/dA_f[t] = dE/dF[t] * F[t] * (1 - F[t]) // due to derative of sigmoid
    // dE/dA_i[t] = dE/dI[t] * dI[t]/dA_i[t] = dE/dI[t] * I[t] * (1 - I[t]) // due to derative of sigmoid
    // dE/dA_c[t] = dE/dC[t] * dC[t]/dA_c[t] = dE/dC[t] * (1 - C[t]*C[t])   // due to derivative of tanh
    // dE/dA_o[t] = dE/dO[t] * dO[t]/dA_o[t] = dE/dO[t] * O[t] * (1 - O[t]) // due to derative of sigmoid

    // Now finally the derivatives wrt. weights connected to previous hidden state
    // dE/dv_f = dE/dA_f[t] · dA_f[t]/dv_f = dE/dA_f[t] · H[t-1]
    // dE/dv_i = dE/dA_i[t] · dA_i[t]/dv_i = dE/dA_i[t] · H[t-1]
    // dE/dv_c = dE/dA_c[t] · dA_c[t]/dv_c = dE/dA_c[t] · H[t-1]
    // dE/dv_o = dE/dA_o[t] · dA_o[t]/dv_o = dE/dA_o[t] · H[t-1]

    // And with respect to weights connected to sequence input
    // dE/dw_f = dE/dA_f[t] · dA_f[t]/dw_f = dE/dA_f[t] · X[t]
    // dE/dw_i = dE/dA_i[t] · dA_f[t]/dw_i = dE/dA_i[t] · X[t]
    // dE/dw_c = dE/dA_c[t] · dA_f[t]/dw_c = dE/dA_c[t] · X[t]
    // dE/dw_o = dE/dA_o[t] · dA_f[t]/dw_o = dE/dA_o[t] · X[t]

    // and bias weights
    // dE/db_f = dE/dA_f[t] · dA_f[t]/db_f = dE/dA_f[t]
    // dE/db_i = dE/dA_i[t] · dA_i[t]/db_i = dE/dA_i[t]
    // dE/db_c = dE/dA_c[t] · dA_c[t]/db_c = dE/dA_c[t]
    // dE/db_o = dE/dA_o[t] · dA_o[t]/db_o = dE/dA_o[t]

    // In practice we calculate dw, dv and db together as one

    // derivative of cell state at time t depends on the derivative at time t+1.
    // we get two contributions, using the formulas for H[t] and S[t]
    // dE/dS[t] = ( dE/dH[t] * dH[t]/dS[t] ) + ( dE/dS[t+1] * dS[t+1]/dS[t] )
    // dE/dS[t] = ( dE/dH[t] * O[t] * (1 - tanh(S[t])^2) ) + ( dE/dS[t+1] * F[t+1] )

    // dE/dH[t] has four contributions, each from the gates depending on H[t-1], along with delta_H[t]
    // dE/dH[t] = dE/dA_f[t+1] * dA_f[t+1]/dH[t] +
    //            dE/dA_i[t+1] * dA_i[t+1]/dH[t] +
    //            dE/dA_c[t+1] * dA_c[t+1]/dH[t] +
    //            dE/dA_o[t+1] * dA_o[t+1]/dH[t] + delta_H[t]
    // therefore we have
    // dE/dH[t] = dE/dA_f[t+1] * v_f + dE/A_i[t+1] * v_i + dE/dA_c[t+1] * v_c + dE/dA_o[t+1] * v_o + delta_H[t]
    // crucially, at last timestep the dE/dA terms are undefined, so only delta_H contributes. And since we only
    // take the last hidden state, delta_H[t] only contributes to the last timestep

    // dE/dw matrices
    float **dw_f = fmatrix(0, p_hs_dim-1, 0, inp_dim + p_hs_dim );
    float **dw_i = fmatrix(0, p_hs_dim-1, 0, inp_dim + p_hs_dim );
    float **dw_c = fmatrix(0, p_hs_dim-1, 0, inp_dim + p_hs_dim );
    float **dw_o = fmatrix(0, p_hs_dim-1, 0, inp_dim + p_hs_dim );

    float *dS_next = fvector(0, p_hs_dim-1); // dE/dS[t+1]
    float *dS = fvector(0, p_hs_dim-1); // dE/dS[t]

    float *delta_H = fvector(0, p_hs_dim-1);
    float *dH = fvector(0, p_hs_dim-1); // dE/dH[t]

    float *dA_f = fvector(0, p_hs_dim-1);
    float *dA_i = fvector(0, p_hs_dim-1);
    float *dA_c = fvector(0, p_hs_dim-1);
    float *dA_o = fvector(0, p_hs_dim-1);
    float *dA_f_next = fvector(0, p_hs_dim-1);
    float *dA_i_next = fvector(0, p_hs_dim-1);
    float *dA_c_next = fvector(0, p_hs_dim-1);
    float *dA_o_next = fvector(0, p_hs_dim-1);

    int Z_len = inp_dim + p_hs_dim; // no bias here
    total_inp = p_hs_dim + inp_dim + 1; // with bias

    k = 0; // overall FFNN input layer neuron counter
    // loop over input chains
    for (i=0; i < n_inp; i++ ) {

        chain = tl->chains[i];

         // loop over LSTMs
        for (l=0; l < p_n_lstm ; l++) { // if bi-lstm, p_n_lstm = p_n_lstm*2

            j = i*p_n_lstm+l;
            m = j * p_hs_dim;

            for (h=0; h < p_hs_dim; h++) { // loop over number of FFNN input neurons derived from this LSTM's last hidden state output

                // calculate delta_H for this neuron based on deltas from previous layer
                delta_H[h] = 0.0;

                for (n = 0; n < s->nnlay[2]; n++) // loop over first FFNN hidden layer neurons
                    delta_H[h] += delta[2][n] * wgt[1][n][k];

                k += 1;

                // reset derivative matrices
                dH[h] = 0.0;
                dS[h] = 0.0;
                dS_next[h] = 0.0;
                dA_f[h] = 0.0;
                dA_i[h] = 0.0;
                dA_c[h] = 0.0;
                dA_o[h] = 0.0;
                dA_f_next[h] = 0.0;
                dA_i_next[h] = 0.0;
                dA_c_next[h] = 0.0;
                dA_o_next[h] = 0.0;

                for (z=0; z < total_inp; z++) { // includes the biases
                    dw_f[h][z] = 0.0;
                    dw_i[h][z] = 0.0;
                    dw_c[h][z] = 0.0;
                    dw_o[h][z] = 0.0;
                }
            }

            // loop over timepoints in reverse order
            // gradients are accumulated through time
            for (t=chain->len-1; t >= 0; t--) {

                for (h=0; h < p_hs_dim; h++) { // dimension of partial derivatives

                    wgt_mh = wgt[0][m + h];

                    // pre-define to avoid accessing too many times
                    S_act = tanh(S[j][t][h]);
                    f_val = F[j][t][h];
                    i_val = I[j][t][h];
                    c_val = C[j][t][h];
                    o_val = O[j][t][h];

                    // first calculate dE/dH[t] and dE/dS[t]
                    if (t == chain->len-1)  {
                        dH[h] = delta_H[h];
                        dS[h] = dH[h] * O[j][t][h] * (1 - S_act*S_act);
                    }
                    else {
                        dH[h] = 0.0;
                        for (z=inp_dim; z < Z_len; z++) // loop over v weights connected to part of Z holding the hidden state (excluding bias)
                            dH[h] +=  dA_f_next[h] * wgt_mh[0 * total_inp + z] +
                                      dA_i_next[h] * wgt_mh[1 * total_inp + z] +
                                      dA_c_next[h] * wgt_mh[2 * total_inp + z] +
                                      dA_o_next[h] * wgt_mh[3 * total_inp + z];

                        dS[h] = dH[h] * o_val * (1 - S_act*S_act) + dS_next[h] * F[j][t+1][h];
                    }

                    if (t > 0)
                    //              dE/dF[t] = dE/dS[t] * S[t-1]
                        dA_f[h] = (dS[h] * S[j][t-1][h]) * f_val * (1 - f_val);
                    else
                        //dA_f[h] = F[j][t][h] * (1 - F[j][t][h]);
                        dA_f[h] = 0.0;

                    //          dE/dI[t] = dE/dS[t] * C[t]
                    dA_i[h] = (dS[h] * c_val) * i_val * (1 - i_val);
                    //          dE/dC[t] = dE/dS[t] * I[t]
                    dA_c[h] = (dS[h] * i_val) * (1 - (c_val * c_val));
                    //         dE/dO[t] = dE/dH[t] * tanh(S[t])
                    dA_o[h] = dH[h] * S_act * o_val * (1 - o_val);

                    // Now calculate dw, dw and db (we store them all in the dw matrices)
                    for (z=0; z < total_inp; z++) { // includes the biases
                        z_val = Z[j][t][z];

                        dw_f[h][z] += dA_f[h] * z_val;
                        dw_i[h][z] += dA_i[h] * z_val;
                        dw_c[h][z] += dA_c[h] * z_val;
                        dw_o[h][z] += dA_o[h] * z_val;
                    }

                    // END: copy dS to dS_next
                    dS_next[h] = dS[h];

                    // copy dA to dA_next
                    dA_f_next[h] = dA_f[h];
                    dA_i_next[h] = dA_i[h];
                    dA_c_next[h] = dA_c[h];
                    dA_o_next[h] = dA_o[h];
                }
            }

            // nin = p_hs_dim * p_n_lstm * n_inp;
            // syn->nnlay[0] = (p_hs_dim * (p_hs_dim + alen + 1)) * 4; // F I C O matrices

            for (h = 0; h < p_hs_dim; h++) {
                wgt_mh = wgt[0][m + h];
                for (z=0; z < total_inp; z++) {  // total_inp includes the bias
                    //printf("%f %f %f %f\n", dw_f[h][z], dw_i[h][z], dw_c[h][z], dw_o[h][z]);
                    //exit(1);

                    if (p_adam) {

                        // F
                        g = dw_f[h][z];
                        moments[m+h][0 * total_inp + z] = p_beta1 * moments[m+h][0 * total_inp + z] + beta1_min * g;
                        velocity[m+h][0 * total_inp + z] = p_beta2 * velocity[m+h][0 * total_inp + z] + beta2_min * g * g;

                        mom_hat = moments[m+h][0 * total_inp + z] / b1p; // bias-corrected estimate
                        vel_hat = velocity[m+h][0 * total_inp + z] / b2p; // bias-corrected estimate

                        wgt_mh[0 * total_inp + z] -= p_etaLSTM * mom_hat / (sqrt(vel_hat) + p_epsilon);

                        // I
                        g = dw_i[h][z];
                        moments[m+h][1 * total_inp + z] = p_beta1 * moments[m+h][1 * total_inp + z] + beta1_min * g;
                        velocity[m+h][1 * total_inp + z] = p_beta2 * velocity[m+h][1 * total_inp + z] + beta2_min * g * g;

                        mom_hat = moments[m+h][1 * total_inp + z] / b1p; // bias-corrected estimate
                        vel_hat = velocity[m+h][1 * total_inp + z] / b2p; // bias-corrected estimate

                        wgt_mh[1 * total_inp + z] -= p_etaLSTM * mom_hat / (sqrt(vel_hat) + p_epsilon);

                        // C
                        g = dw_c[h][z];
                        moments[m+h][2 * total_inp + z] = p_beta1 * moments[m+h][2 * total_inp + z] + beta1_min * g;
                        velocity[m+h][2 * total_inp + z] = p_beta2 * velocity[m+h][2 * total_inp + z] + beta2_min * g * g;

                        mom_hat = moments[m+h][2 * total_inp + z] / b1p; // bias-corrected estimate
                        vel_hat = velocity[m+h][2 * total_inp + z] / b2p; // bias-corrected estimate

                        wgt_mh[2 * total_inp + z] -= p_etaLSTM * mom_hat / (sqrt(vel_hat) + p_epsilon);

                        // O
                        g = dw_o[h][z];
                        moments[m+h][3 * total_inp + z] = p_beta1 * moments[m+h][3 * total_inp + z] + beta1_min * g;
                        velocity[m+h][3 * total_inp + z] = p_beta2 * velocity[m+h][3 * total_inp + z] + beta2_min * g * g;

                        mom_hat = moments[m+h][3 * total_inp + z] / b1p; // bias-corrected estimate
                        vel_hat = velocity[m+h][3 * total_inp + z] / b2p; // bias-corrected estimate

                        wgt_mh[3 * total_inp + z] -= p_etaLSTM * mom_hat / (sqrt(vel_hat) + p_epsilon);

                    } else {
                        wgt_mh[0 * total_inp + z] -= dw_f[h][z] * p_etaLSTM;
                        wgt_mh[1 * total_inp + z] -= dw_i[h][z] * p_etaLSTM;
                        wgt_mh[2 * total_inp + z] -= dw_c[h][z] * p_etaLSTM;
                        wgt_mh[3 * total_inp + z] -= dw_o[h][z] * p_etaLSTM;
                    }
                }
            }
        }
    }

    fmatrix_free(delta, 0, s->nlay-1, 0, s->maxnper);

    fmatrix_free(dw_f, 0, p_hs_dim-1, 0, inp_dim + p_hs_dim );
    fmatrix_free(dw_i, 0, p_hs_dim-1, 0, inp_dim + p_hs_dim );
    fmatrix_free(dw_c, 0, p_hs_dim-1, 0, inp_dim + p_hs_dim );
    fmatrix_free(dw_o, 0, p_hs_dim-1, 0, inp_dim + p_hs_dim );

    fvector_free(dS_next, 0, p_hs_dim-1); // dE/dS[t+1]
    fvector_free(dS, 0, p_hs_dim-1); // dE/dS[t]

    fvector_free(delta_H, 0, p_hs_dim-1);
    fvector_free(dH, 0, p_hs_dim-1); // dE/dH[t]

    fvector_free(dA_f, 0, p_hs_dim-1);
    fvector_free(dA_i, 0, p_hs_dim-1);
    fvector_free(dA_c, 0, p_hs_dim-1);
    fvector_free(dA_o, 0, p_hs_dim-1);

    fvector_free(dA_f_next, 0, p_hs_dim-1);
    fvector_free(dA_i_next, 0, p_hs_dim-1);
    fvector_free(dA_c_next, 0, p_hs_dim-1);
    fvector_free(dA_o_next, 0, p_hs_dim-1);

}

void get_score(float *score_test, TCRLIST *testlist, SYNAPS *syn, float *inp, int n_inp, int inp_dim,
    float ***Z, float ***F, float ***I, float ***C, float ***O, float ***S, float ***H, int alen, float **x, float **blmat, int outlay, int *inv_vector)

{

    float max_score;
    int i, n, best_idx;
    TCRLIST *tl;

    for (i=0,tl=testlist; tl;tl=tl->next, i++) {

        forward_lstm(tl, syn, inp, n_inp, inp_dim, Z, F, I, C, O, S, H, NULL, inv_vector);

        forward_inp(x, syn, inp, alen, 0);

        score_test[i] = x[outlay][0];
    }
}

float ***init_wgt(SYNAPS *syn)

{
    float ***wgt;
    int l, k, i;

    wgt = (float ***) malloc((unsigned)(syn->nlay - 1) * sizeof(float **));

    if (!wgt) {
        printf("Allocation failure 1 in synaps_wmtx_alloc\n");
        exit(1);
    }

    for (l = 0; l < syn->nlay - 1; l++) {

        wgt[l] = fmatrix(0, syn->nnlay[l + 1] - 1, 0, syn->nnlay[l]);

        if (!wgt[l]) {
            printf("Allocation failure 2 in synaps_wmtx_alloc\n");
            exit(1);
        }

        for (k = 0; k < syn->nnlay[l + 1]; k++)
            for (i = 0; i < syn->nnlay[l] + 1; i++) {
                if (p_initweight)
                    wgt[l][k][i] = p_weight * (0.5 - drand48());
                else
                    wgt[l][k][i] = (drand48() > 0.5 ? p_weight : -p_weight);

                if (p_useRELU_initweight && l == -1)
                    wgt[l][k][i] *= 1.0 / sqrt(syn->nnlay[l] + 1);
            }
    }

    return (wgt);
}

void syn_print(SYNAPS *syn, int cycle, char *filename, char *comment, char *options) {
    FILE * fp;
    int l, i, k, nc;

    if ((fp = fopen(filename, "w")) == NULL) {
        printf("Cannot open file %s\n", filename);
        exit(1);
    }

    fprintf(fp, "TESTRUNID %s %s\n", comment, options);

    for (l = 0; l < syn->nlay; l++)
        fprintf(fp, "%8i  LAYER: %4i\n", syn->nnlay[l], l + 1);

    fprintf(fp, "%8i   :ILEARN\n", cycle);

    nc = 0;

    for (l = 0; l < syn->nlay - 1; l++)
        for (k = 0; k < syn->nnlay[l + 1]; k++) // no nin_pep as all inputs come from CNN
            for (i = 0; i < syn->nnlay[l] + 1; i++) {
                fprintf(fp, "%13f", syn->wgt[l][k][i]);

                nc++;

                if (nc == 5) {
                    fprintf(fp, "\n");
                    nc = 0;
                }
            }

    if (nc != 0)
        fprintf(fp, "\n");

    fclose(fp);

}


int main(int argc, char *argv[])

{

    int alen;
    int i, j, n, l, m, h;
    int iter, best_iter;
    int ix;
    LINE com;
    float *score, *aff;
    float *score_test, *aff_test, tcc, best_tcc, ter, best_ter;
    int nseq, nseqtest;
    float fcc, ferr;
    LINE buff, bestbuff, optionbuff, filter_buffer;
    int nb;
    FILENAME filename;
    float **blmat;
    float *inp;
    int *order;
    SYNAPS *syn;
    float **x;
    int nweigths, nweigths_lstm;
    int *nhid, nhlay;
    int *inv_vector;

    float ***Z, ***F, ***I, ***C, ***O, ***S, ***H; // LSTM matrices
    int inp_dim;

    int n_inp = 0;
    int current_nfilters = 0;
    int max_nfilters = 0;
    int inp_idx, best_idx;
    int maxnper;
    int nin, nout, outlay;
    int *max_seq_len;
    float auc, auc_t;
    int nc;

    float **moments_table;
    float **velocity_table;
    float *beta1_pow, beta1_min;
    float *beta2_pow, beta2_min;

    TCRLIST *trainlist, *tl, *testlist = NULL;
    TCRLIST **tcr_table;

    float *dropout_mask = NULL;

    com[0] = 0;
    for (i = 0; i < argc; i++) {
        strcat(com, argv[i]);
        strcat(com, " ");
    }

    pparse( &argc, &argv, param, 1, "trainfile");


    if (p_seed >= 0)
        setseed(p_seed);

    if (p_bl) {
        sprintf(filename, p_blosummat, p_bls);
        bl_init_file(filename);

        printf("# Blosum matrix %i initialized\n", p_bls);

        blmat = fmatrix(0, 19, 0, 19);
        for (i = 0; i < 20; i++)
            for (j = 0; j < 20; j++)
                blmat[i][j] = (float)(bl_s(i, j) / p_blnorm);
    }

    alen = strlen(p_alphabet);

    if (strcmp(p_nhid, "UNDEF") != 0) {
        nhid = ivector_split(p_nhid, &nhlay);
    }
    else {
        nhid = ivector_init(0,0,1);
        nhlay = 1;
    }

    for (i = 0; i < nhlay; i++)
        printf("# Hiden layer size %i %i\n", i, nhid[i]);

    if (p_sampleweight)
        trainlist = read_tcrlist_wweights(argv[1]);
    else
        trainlist = read_tcrlist(argv[1]);

    if (trainlist == NULL) {
        printf("Error. No elements in tcrlist\n");
        exit(1);
    }

    n_inp = trainlist->n_inp;

    if (strlen(p_testfile) > 0) {

        if (p_sampleweight)
            testlist = read_tcrlist_wweights(p_testfile);
        else
            testlist = read_tcrlist(p_testfile);

        if (testlist == NULL) {
            printf("Error. No elements in tcrlist\n");
            exit(1);
        }
        if (testlist->n_inp != n_inp) {
            printf("Error: mismatch between testlist n_inp and trainlist n_inp %i %i\n",  testlist->n_inp, n_inp);
            exit(1);
        }
    }

    max_seq_len = ivector_init(0, n_inp-1, -99);

    // get maximum sequence length per input
    for (tl=trainlist; tl;tl=tl->next)
        for (i=0; i < n_inp; i++)
            if ((tl->chains[i])->len > max_seq_len[i])
                max_seq_len[i] = (tl->chains[i])->len;

    if (testlist)
        for (tl=testlist; tl;tl=tl->next)
            for (i=0; i < n_inp; i++)
                if ((tl->chains[i])->len > max_seq_len[i])
                    max_seq_len[i] = (tl->chains[i])->len;


    for (i=0; i < n_inp; i++)
        printf("# Maximum seq length %i %i\n", i, max_seq_len[i]);

    sprintf(optionbuff, "OPTIONS N_INP: %i N_LSTM: %i HS_DIM: %i BILSTM: %i", n_inp, p_n_lstm, p_hs_dim, p_bilstm);

    // Allocate LSTM matrices

    inp_dim = alen; // sequence input dimension per timestep (i.e. per sequence position)

    if (p_bilstm)
        p_n_lstm *= 2;

    Z = (float ***) malloc((unsigned)(n_inp * p_n_lstm) * sizeof(float **));
    F = (float ***) malloc((unsigned)(n_inp * p_n_lstm) * sizeof(float **));
    I = (float ***) malloc((unsigned)(n_inp * p_n_lstm) * sizeof(float **));
    C = (float ***) malloc((unsigned)(n_inp * p_n_lstm) * sizeof(float **));
    O = (float ***) malloc((unsigned)(n_inp * p_n_lstm) * sizeof(float **));
    S = (float ***) malloc((unsigned)(n_inp * p_n_lstm) * sizeof(float **));
    H = (float ***) malloc((unsigned)(n_inp * p_n_lstm) * sizeof(float **));

    inv_vector = ivector(0, n_inp * p_n_lstm-1);

    // Z[i*p_n_lstm+l][t][z]; concatenated lstm input
    // F[i*p_n_lstm+l][t][h] F I C O S H
    for (i=0; i < n_inp; i++) {

        for (l=0; l < p_n_lstm ; l++) {

            Z[i*p_n_lstm + l] = fmatrix(0, max_seq_len[i]-1, 0, inp_dim + p_hs_dim);

            F[i*p_n_lstm + l] = fmatrix(0, max_seq_len[i]-1, 0, p_hs_dim - 1);
            I[i*p_n_lstm + l] = fmatrix(0, max_seq_len[i]-1, 0, p_hs_dim - 1);
            C[i*p_n_lstm + l] = fmatrix(0, max_seq_len[i]-1, 0, p_hs_dim - 1);
            O[i*p_n_lstm + l] = fmatrix(0, max_seq_len[i]-1, 0, p_hs_dim - 1);
            S[i*p_n_lstm + l] = fmatrix(0, max_seq_len[i]-1, 0, p_hs_dim - 1);
            H[i*p_n_lstm + l] = fmatrix(0, max_seq_len[i]-1, 0, p_hs_dim - 1);

            if (p_bilstm)
                inv_vector[i*p_n_lstm + l] = l % 2;
            else
                inv_vector[i*p_n_lstm + l] = 0;
        }
    }
    sprintf(buff, "%i", p_niter);

    // encode sequences
    tcrlist2encode_chain(trainlist, alen, blmat);

    for (nseq = 0, tl = trainlist; tl; tl = tl->next, nseq++);
    printf("# Number of training sequences %i\n", nseq);

    aff = fvector(0, nseq - 1);
    score = fvector(0, nseq - 1);

    for (j = 0, tl = trainlist; tl; tl = tl->next, j++)
        aff[j] = tl->aff;

    if (testlist) {

        for (nseqtest = 0, tl = testlist; tl; tl = tl->next, nseqtest++);

        printf("# Number of testset data %i\n", nseqtest);

        score_test = fvector(0, nseqtest - 1);
        aff_test = fvector(0, nseqtest - 1);

        for (j = 0, tl = testlist; tl; tl = tl->next, j++)
            aff_test[j] = tl->aff;

        tcrlist2encode_chain(testlist, alen, blmat);
    }

    if ((syn = synaps_alloc()) == NULL) {
        printf("Error. Cannot allocate SYNAPS\n");
        exit(1);
    }

    nout = 2;
    nin = p_hs_dim * p_n_lstm * n_inp; // number of LSTM outputs

    syn->nlay = 3 + nhlay;
    syn->nnlay = ivector(0, syn->nlay - 1);

    outlay = syn->nlay - 1;

    maxnper = nin;
    syn->nnlay[0] = (p_hs_dim + inp_dim + 1) * 4; // F I C O matrices - number of weights for each LSTM, without the factor of p_hs_dim included in nin
    syn->nnlay[1] = nin; // input layer to FFNN
    for (i=0; i < nhlay; i++) {
        syn->nnlay[2+i] = nhid[i];

        if (nhid[i] > maxnper)
            maxnper = nhid[i];
    }
    syn->nnlay[2+nhlay] = nout;

    syn->maxnper = maxnper;

    printf("# Ninput %i. Noutput %i\n", nin, nout);

    for (i = 0; i < syn->nlay; i++)
        printf("# Nlayer %i NNlay %i\n", i, syn->nnlay[i]);

    nweigths = 0;
    for (i=2; i < syn->nlay; i++)
        nweigths += syn->nnlay[i] * syn->nnlay[i-1] + syn->nnlay[i];
    nweigths_lstm = p_n_lstm * n_inp * (p_hs_dim + inp_dim + 1) * p_hs_dim;

    printf( "# Number of weights in FFNN network %i. Weights LSTM %i Weights total %i\n", nweigths, nweigths_lstm, nweigths + nweigths_lstm );

    // x contains output values per neuron per layer
    x = fmatrix(1, syn->nlay - 1, 0, syn->maxnper); // Used for FFNN. Starts at 1 because the 0-th layer is the LSTM layer

    syn->wgt = init_wgt(syn);

    // set forget biases to a high number, improves performance
    // https://github.com/mlresearch/v37/blob/gh-pages/jozefowicz15.pdf
    for (i = 0; i < n_inp; i++) {
        for (l = 0; l < p_n_lstm; l++) {
            j = i * p_n_lstm + l;
            m = j * p_hs_dim;

            for (h = 0; h < p_hs_dim; h++) {
                syn->wgt[0][m + h][0*(inp_dim + p_hs_dim+1) + inp_dim + p_hs_dim] = 2.0; // forget gate bias
                //syn->wgt[0][m + h][1*(inp_dim + p_hs_dim+1) + inp_dim + p_hs_dim] = -2.0; // input gate bias
                //syn->wgt[0][m + h][2*(inp_dim + p_hs_dim+1) + inp_dim + p_hs_dim] = -2.0; // input gate bias
                //syn->wgt[0][m + h][3*(inp_dim + p_hs_dim+1) + inp_dim + p_hs_dim] = -2.0; // input gate bias
            }
        }
    }

    tcr_table = (TCRLIST **) malloc(( unsigned ) nseq * sizeof( TCRLIST * ));

    for (i = 0, tl = trainlist; tl; tl = tl->next, i++)
        tcr_table[i] = tl;

    order = ivector_ramp(0, nseq - 1);
    inp = fvector(0, nin - 1);

    // For Adam optimizer
    moments_table = fmatrix(0, p_hs_dim * p_n_lstm * n_inp - 1, 0, (p_hs_dim + inp_dim + 1) * 4);
    velocity_table = fmatrix(0, p_hs_dim * p_n_lstm * n_inp - 1, 0, (p_hs_dim + inp_dim + 1) * 4);

    beta1_pow = fvector(0, p_niter - 1);
    beta2_pow = fvector(0, p_niter - 1);

    // precompute powers of beta1 and beta2 for adam
    for (i=0; i < p_niter; i++) {
        beta1_pow[i] = 1.0 - powf(p_beta1, i+1);
        beta2_pow[i] = 1.0 - powf(p_beta2, i+1);
    }
    beta1_min = 1.0 - p_beta1;
    beta2_min = 1.0 - p_beta2;

    if (p_dropout_lstm > -1) {
        dropout_mask = fvector(0, inp_dim + p_hs_dim - 1);
    }

    best_ter = 9999.9;
    best_tcc = -9999.9;

    for (iter = 0; iter < p_niter; iter++) {

        ivector_rerandomize(order, 0, nseq - 1);

        for (i = 0; i < nseq; i++) {

            ix = order[i];
            tl = tcr_table[ix];

            forward_lstm(tl, syn, inp, n_inp, inp_dim, Z, F, I, C, O, S, H, dropout_mask, inv_vector);

            forward_inp(x, syn, inp, alen, 1);

            score[ix] = x[outlay][0];
            aff[ix] = tl->aff;

            /*if (iter >= p_burnin) {

                if ((tl->aff > 0.5 && x[outlay][0] < 0.1) || (tl->aff < 0.5 && x[outlay][0] > 0.9))
                    continue;
            }*/

            backprob(tl, x, syn, tl->aff, tl->w, n_inp, inp_dim, Z, F, I, C, O, S, H,  moments_table, velocity_table, beta1_pow[iter], beta2_pow[iter], beta1_min, beta2_min);

        }
        auc = cal_auc_frac(aff, score, nseq, 0.5, 1.0 );
        fcc = cal_auc_frac(aff, score, nseq, 0.5, 0.1 );
        ferr = fvector_xyerror(nseq, score, aff);
        printf("# Iter %i Train AUC %f AUC_01 %f Err %f ", iter, auc, fcc, ferr);

        if (testlist && (iter % p_ntest) == 0) {

            get_score(score_test, testlist, syn, inp, n_inp, inp_dim, Z, F, I, C, O, S, H, alen, x, blmat, outlay, inv_vector);

            auc_t = cal_auc_frac(aff_test, score_test, nseqtest, 0.5, 1.0 );
            tcc = cal_auc_frac(aff_test, score_test, nseqtest, 0.5, 0.1 );
            ter = fvector_xyerror(nseqtest, score_test, aff_test);

            printf("Test AUC %f AUC_01 %f Test Err %f ", auc_t,  tcc, ter);

            if ( p_teststop && tcc > best_tcc) {

                best_ter = ter;
                best_tcc  = tcc;
                best_iter = iter;

                sprintf( bestbuff, "Ncycles %i Train AUC %f AUC_01 %f Err %f Test AUC %f AUC_01 %f Err %f", iter, auc, fcc, ferr, auc_t, tcc, ter );

                syn_print( syn, iter, p_syn, bestbuff, optionbuff);

                printf( "# Dump synaps");

                //printf("<- Best Test AUC_01");
            }
        }
        printf("\n");

    }

    exit(0);
}
