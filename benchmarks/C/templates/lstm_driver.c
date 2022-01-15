#include "lstm.h"

int main() {
  LSTMInput input;
  read_lstm_instance(&input);
  printf("%d %d %d\n", input.l, input.c, input.b);
}
