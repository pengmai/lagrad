#include <math.h>
#include <stdlib.h>

extern int enzyme_const;
extern int enzyme_dup;
extern int enzyme_dupnoneed;
void __enzyme_autodiff(void *, ...);

double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

// log(sum(exp(x), 2))
double logsumexp(double const *vect, int sz) {
  double sum = 0.0;
  int i;

  for (i = 0; i < sz; i++) {
    sum += exp(vect[i]);
  }

  sum += 2;
  return log(sum);
}

// LSTM OBJECTIVE
// The LSTM model
void lstm_model(int hsize, double const *__restrict weight,
                double const *__restrict bias, double *__restrict hidden,
                double *__restrict cell, double const *__restrict input) {
  // TODO NOTE THIS
  // __builtin_assume(hsize > 0);

  double *gates = (double *)malloc(4 * hsize * sizeof(double));
  double *forget = &(gates[0]);
  double *ingate = &(gates[hsize]);
  double *outgate = &(gates[2 * hsize]);
  double *change = &(gates[3 * hsize]);

  int i;
  // caching input
  // hidden (needed)
  for (i = 0; i < hsize; i++) {
    forget[i] = sigmoid(input[i] * weight[i] + bias[i]);
    ingate[i] = sigmoid(hidden[i] * weight[hsize + i] + bias[hsize + i]);
    outgate[i] =
        sigmoid(input[i] * weight[2 * hsize + i] + bias[2 * hsize + i]);
    change[i] = tanh(hidden[i] * weight[3 * hsize + i] + bias[3 * hsize + i]);
  }

  // caching cell (needed)
  for (i = 0; i < hsize; i++) {
    cell[i] = cell[i] * forget[i] + ingate[i] * change[i];
  }

  for (i = 0; i < hsize; i++) {
    hidden[i] = outgate[i] * tanh(cell[i]);
  }

  free(gates);
}

// Predict LSTM output given an input
void lstm_predict(int l, int b, double const *__restrict w,
                  double const *__restrict w2, double *__restrict s,
                  double const *__restrict x, double *__restrict x2) {
  int i;
  for (i = 0; i < b; i++) {
    x2[i] = x[i] * w2[i];
  }

  double *xp = x2;
  for (int raw_i = 0; raw_i < l; raw_i++) {
    i = raw_i * 2 * b;
    lstm_model(b, /*weight=*/&(w[i * 4]), /*bias=*/&(w[(i + b) * 4]),
               /*hidden=*/&(s[i]), /*cell=*/&(s[i + b]), /*input=*/xp);
    xp = &(s[i]);
  }

  for (i = 0; i < b; i++) {
    x2[i] = xp[i] * w2[b + i] + w2[2 * b + i];
  }
}

// LSTM objective (loss function)
void lstm_objective(int l, int c, int b, double const *__restrict main_params,
                    double const *__restrict extra_params,
                    double *__restrict state, double const *__restrict sequence,
                    double *__restrict loss) {
  int i, t;
  double total = 0.0;
  int count = 0;
  const double *input = &(sequence[0]);
  double *ypred = (double *)malloc(b * sizeof(double));
  double *ynorm = (double *)malloc(b * sizeof(double));
  const double *ygold;
  double lse;

  // __builtin_assume(b > 0);
  // for (t = 0; t <= (c - 1) * b - 1; t += b) {
  for (int rawt = 0; rawt < c - 1; rawt++) {
    t = rawt * b;
    lstm_predict(l, b, main_params, extra_params, state, input, ypred);

    lse = logsumexp(ypred, b);
    for (i = 0; i < b; i++) {
      ynorm[i] = ypred[i] - lse;
    }

    ygold = &(sequence[t + b]);
    for (i = 0; i < b; i++) {
      total += ygold[i] * ynorm[i];
    }

    count += b;
    input = ygold;
  }

  *loss = -total / count;

  free(ypred);
  free(ynorm);
}

void enzyme_c_lstm_objective(int l, int c, int b, double const *main_params,
                             double *dmain_params, double const *extra_params,
                             double *dextra_params, double *state,
                             double const *sequence, double *loss,
                             double *dloss) {
  double *dstate = calloc(2 * l * b, sizeof(double));
  __enzyme_autodiff(lstm_objective, enzyme_const, l, enzyme_const, c,
                    enzyme_const, b, enzyme_dup, main_params, dmain_params,
                    enzyme_dup, extra_params, dextra_params, enzyme_dup, state,
                    dstate, enzyme_const, sequence, enzyme_dupnoneed, loss,
                    dloss);
  free(dstate);
}
