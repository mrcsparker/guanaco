require 'guanaco'

camel = Guanaco::Camel::from_pretrained("./cache/llama2_7b_chat_uncensored.ggmlv3.q4_0.bin")

output = camel.generate("The meaning of life is")
puts output
