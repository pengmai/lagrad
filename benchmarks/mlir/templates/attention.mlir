// An implementation of scaled dot product attention used in a Transformer decoder.

func @scaled_dot_attention(
  %Q: tensor<{{h}}x{{h}}xf64>,
  %K: tensor<{{h}}x{{h}}xf64>,
  %V: tensor<{{h}}x{{h}}xf64>,
  %mlp_weights: tensor<{{h}}x{{h}}xf64>,
  %mlp_biases: tensor<{{h}}xf64>,
  %queries: tensor<{{b}}x{{s}}x{{h}}xf64>,
  %keys: tensor<{{b}}x{{s}}x{{h}}xf64>,
  %values: tensor<{{b}}x{{s}}x{{h}}xf64>
) {
  return
}
