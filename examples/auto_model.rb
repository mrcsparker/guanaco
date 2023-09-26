require 'guanaco'

auto_model = Guanaco::AutoModel::from_pretrained("./cache/llama2_7b_chat_uncensored.ggmlv3.q4_0.bin")

output = auto_model.generate("The quick brown fox jumped over the")
puts output

output = auto_model.generate("What is the weather like in New York?")
puts output