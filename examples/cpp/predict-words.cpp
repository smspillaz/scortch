#include <cxxopts.hpp>

#include <torch/torch.h>
#include <torch/script.h>

#include <algorithm>
#include <array>
#include <iterator>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>

static char const *test_sentence = \
  "At this point, we have seen various feed-forward networks.\n"
  "That is, there is no state maintained by the network at all.\n"
  "This might not be the behavior we want.\n"
  "Sequence models are central to NLP: they are models where there is some sort\n"
  "of dependence through time between your inputs. The classical example of\n"
  "a sequence model is the Hidden Markov Model for part-of-speech tagging.\n"
  "Another example is the conditional random field.\n" \
  "\n"
  "A recurrent neural network is a network that maintains some kind of state.\n"
  "For example, its output could be used as part of the next input, so\n"
  "that information can propogate along as the network passes over the\n"
  "sequence. In the case of an LSTM, for each element in the sequence,\n"
  "there is a corresponding hidden state ht, which in principle can contain\n"
  "information from arbitrary points earlier in the sequence. We can use\n"
  "the hidden state to predict words in a language model, part-of-speech tags,\n"
  "and a myriad of other things.";

namespace {
  const std::vector<std::string>
  split_string(std::string const &str, char delimiter) {
    std::vector<std::string> words;
    std::string item;
    std::stringstream ss(str);

    while (std::getline(ss, item, delimiter)) {
      words.push_back(item);
    }

    return words;
  }

  template<int N>
  const std::vector<std::tuple<std::string, std::array<std::string, N * 2>>>
  make_context_vector(std::vector <std::string> const &words) {
    std::vector <std::tuple<std::string, std::array<std::string, N * 2> > > contexts;

    for (size_t i = N; i < words.size() - N; ++i) {
      std::array<std::string, N * 2> context;

      for (size_t j = 1; j < N + 1; ++j) {
        context[((N + 1) - j) - 1] = words[i - j];
      }

      for (size_t j = 1; j < N + 1; ++j) {
        context[N + j - 1] = words[i + j];
      }

      contexts.emplace_back(std::make_tuple(words[i], std::move(context)));
    }

    return contexts;
  }

  const std::unordered_map<std::string, long long>
  make_dictionary(std::vector<std::string> const &words) {
    std::unordered_map<std::string, long long> dictionary;
    long long count = 0;

    for (size_t i = 0; i < words.size(); ++i) {
      if (dictionary.find(words[i]) == dictionary.end()) {
        dictionary[words[i]] = count++;
      }
    }

    return dictionary;
  }

  struct CBOWLanguageModeller: torch::nn::Module {
    CBOWLanguageModeller(size_t vocab_size,
                         size_t embedding_dim,
                         size_t context_size) :
      embedding(register_module("embedding",
                                torch::nn::Embedding(vocab_size, embedding_dim))),
      fc1(register_module("fc1", torch::nn::Linear(embedding_dim * context_size * 2,
                                                   128))),
      fc2(register_module("fc2", torch::nn::Linear(128, vocab_size)))
    {
    }

    // Implement the Net's algorithm.
    torch::Tensor forward(torch::Tensor x) {
      x = torch::relu(fc1->forward(embedding->forward(x).view({1, -1})));
      return torch::log_softmax(fc2->forward(x), 1);
    }

    torch::nn::Embedding embedding{nullptr};
    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
  };

  template <typename WordArray>
  std::vector<long long>
  words_to_indices(std::unordered_map<std::string, long long> const &vocab,
                   WordArray const &words) {
    std::vector<long long> indices;
    indices.reserve(words.size());

    for (auto const &word : words) {
      indices.push_back(vocab.find(word)->second);
    }

    return indices;
  }

  template<typename T>
  torch::Tensor
  make_tensor_from_vector(std::vector<T> const &vec) {
    auto tensor = torch::full({static_cast<long long>(vec.size())}, 0, torch::kInt64);

    for (size_t i = 0; i < vec.size(); ++i) {
      tensor[i] = torch::Scalar(vec[i]);
    }

    return tensor;
  }

  template<typename T>
  std::vector<T>
  scalar_to_vector(T const &t) {
    std::vector<T> v;
    v.emplace_back(t);
    return v;
  }

  template<typename Context>
  void train_cbow_language_modeller(CBOWLanguageModeller &model,
                                    std::unordered_map<std::string, long long> const &vocab,
                                    Context const &context,
                                    size_t epochs,
                                    float learning_rate) {
    torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(learning_rate));

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
      for (auto const &tuple : context) {
        auto word(make_tensor_from_vector(scalar_to_vector(vocab.find(std::get<0>(tuple))->second)));
        auto context_indices(make_tensor_from_vector(words_to_indices(vocab, std::get<1>(tuple))));

        optimizer.zero_grad();
        auto prediction = model.forward(context_indices);
        auto loss = torch::nll_loss(prediction, word);
        loss.backward();
        optimizer.step();

        std::cout << "Epoch: " << epoch << " loss: " << *loss.template data<float>() << std::endl;
      }
    }
  }

  template <typename Map>
  std::unordered_map<typename Map::mapped_type, typename Map::key_type>
  reverse_map(Map const &m) {
    std::unordered_map<typename Map::mapped_type, typename Map::key_type> reversed;

    for (auto pair : m) {
      reversed[pair.second] = pair.first;
    }

    return reversed;
  }

  template <typename Context>
  std::tuple<std::string, float>
  predict_word(CBOWLanguageModeller &model,
               std::unordered_map<std::string, long long> const &in_vocab,
               std::unordered_map<long long, std::string> const &out_vocab,
               Context const &context) {
    torch::NoGradGuard guard{};
    auto context_indices(make_tensor_from_vector(words_to_indices(in_vocab,
                                                                  std::get<1>(context))));
    auto prediction = model.forward(context_indices);
    torch::Tensor value, index;

    std::tie(value, index) = torch::max(torch::exp(prediction), 1);

    return std::make_tuple(out_vocab.find(*(index.template data<long long>()))->second,
                           *value.template data<float>());
  }

  template <typename Context>
  std::string
  format_word_prediction_for(CBOWLanguageModeller &model,
                             std::unordered_map<std::string, long long> const &in_vocab,
                             std::unordered_map<long long, std::string> const &out_vocab,
                             Context const &context)
  {
    std::stringstream ss;
    std::string word;
    float probability;

    std::tie(word, probability) = predict_word(model,
                                               in_vocab,
                                               out_vocab,
                                               context);

    ss << word << " (" << probability << ")";

    return ss.str();
  }
}

int main(int argc, char **argv) {
  // Construct vocabulary
  auto words = split_string(std::string(test_sentence), ' ');
  auto context = make_context_vector<2>(words);
  auto vocab = make_dictionary(words);
  auto indices_to_words = reverse_map(vocab);

  // Create a new Net.
  CBOWLanguageModeller model(vocab.size(), 10, 2);

  // Instantiate an SGD optimization algorithm to update our Net's parameters.
  torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(0.1));

  train_cbow_language_modeller(model, vocab, context, 50, 0.01);

  for (size_t i = 2; i < words.size() - 2; ++i) {
    std::cout << format_word_prediction_for(model,
                                            vocab,
                                            indices_to_words,
                                            context[i - 2]) << " ";
  }

  std::cout << "\n";

  return 0;
}
