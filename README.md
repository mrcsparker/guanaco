# Guanaco

Run local LLMs in Ruby.

## Usage

### Running local GGML models

Models can be loaded via the AutoModel interface.

```ruby
require 'guanaco'

# load the model
camel = Guanaco::Camel::from_pretrained("./cache/llama2_7b_chat_uncensored.ggmlv3.q4_0.bin")

# generate
output = camel.generate("The meaning of life is")

puts output
```

## Development

After checking out the repo, run `bin/setup` to install dependencies. Then, run `rake spec` to run the tests. You can also run `bin/console` for an interactive prompt that will allow you to experiment.

To install this gem onto your local machine, run `bundle exec rake install`. To release a new version, update the version number in `version.rb`, and then run `bundle exec rake release`, which will create a git tag for the version, push git commits and the created tag, and push the `.gem` file to [rubygems.org](https://rubygems.org).

## Compiling

```sh
> bundle install
> bundle exec rake compile
```

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/mrcsparker/guanaco.

## License

The gem is available as open source under the terms of the [MIT License](https://opensource.org/licenses/MIT).
