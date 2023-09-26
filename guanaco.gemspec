# frozen_string_literal: true

require_relative "lib/guanaco/version"

Gem::Specification.new do |spec|
  spec.name = "guanaco"
  spec.version = Guanaco::VERSION
  spec.authors = ["Chris Parker"]
  spec.email = ["mrcsparker@gmail.com"]

  spec.summary = "LLMs in Ruby"
  spec.description = "LLMs in Ruby"
  spec.homepage = "https://github.com/mrcsparker/guanaco"
  spec.license = "MIT"
  spec.required_ruby_version = ">= 2.6.0"
  spec.required_rubygems_version = ">= 3.3.11"

  spec.metadata["allowed_push_host"] = "https://github.com/mrcsparker/guanaco"

  spec.metadata["homepage_uri"] = spec.homepage
  spec.metadata["source_code_uri"] = "https://github.com/mrcsparker/guanaco"
  spec.metadata["changelog_uri"] = "https://github.com/mrcsparker/guanaco"

  # Specify which files should be added to the gem when it is released.
  # The `git ls-files -z` loads the files in the RubyGem that have been added into git.
  spec.files = Dir.chdir(__dir__) do
    `git ls-files -z`.split("\x0").reject do |f|
      (File.expand_path(f) == __FILE__) || f.start_with?(*%w[bin/ test/ spec/ features/ .git .circleci appveyor])
    end
  end
  spec.bindir = "exe"
  spec.executables = spec.files.grep(%r{\Aexe/}) { |f| File.basename(f) }
  spec.require_paths = ["lib"]
  spec.extensions = ["ext/guanaco/Cargo.toml"]

  # Uncomment to register a new dependency of your gem
  # spec.add_dependency "example-gem", "~> 1.0"

  # For more information and examples about making a new gem, check out our
  # guide at: https://bundler.io/guides/creating_gem.html
end
