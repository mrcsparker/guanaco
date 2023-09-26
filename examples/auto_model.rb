require 'guanaco'

auto_model = Guanaco::AutoModel::from_pretrained("./cache/llama2_7b_chat_uncensored.ggmlv3.q4_0.bin")

output = auto_model.generate("The meaning of life is")
puts output
