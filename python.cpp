#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For std::vector, std::unique_ptr

#include <speex_preprocess.h>

#include <vector>   // For std::vector
#include <string>   // For std::to_string
#include <stdexcept> // For std::runtime_error, std::invalid_argument
#include <cstring>  // For memcpy
#include <memory>   // For std::unique_ptr, std::make_unique

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

// ----------------------------------------------------------------------------

struct ProcessedAudioChunk {
  py::bytes audio; // Holds the processed audio data

  // Constructor that takes an already processed py::bytes object
  ProcessedAudioChunk(py::bytes data) : audio(std::move(data)) {}
};

class AudioProcessor {
private:
  SpeexPreprocessState *state = NULL;
  int m_chunk_size_samples; // Number of samples per chunk
  int m_chunk_size_bytes;   // Number of bytes per chunk (samples * 2 for int16_t)

public:
  AudioProcessor(int chunk_size_samples, float auto_gain,
                 int noise_suppression);
  ~AudioProcessor();

  std::unique_ptr<ProcessedAudioChunk> ProcessChunk(py::bytes audio_input);
};

AudioProcessor::AudioProcessor(int chunk_size_samples, float auto_gain,
                               int noise_suppression)
    : m_chunk_size_samples(chunk_size_samples),
      m_chunk_size_bytes(chunk_size_samples * 2) { // Assuming 2 bytes per sample (int16_t)

  if (chunk_size_samples <= 0) {
    throw std::invalid_argument("chunk_size_samples must be positive.");
  }

  this->state = speex_preprocess_state_init(m_chunk_size_samples, 16000);
  if (!this->state) {
    throw std::runtime_error("Failed to initialize Speex preprocessor state.");
  }

  int noise_state = (noise_suppression == 0) ? 0 : 1;
  speex_preprocess_ctl(state, SPEEX_PREPROCESS_SET_DENOISE, &noise_state);

  if (noise_suppression != 0) {
    speex_preprocess_ctl(state, SPEEX_PREPROCESS_SET_NOISE_SUPPRESS,
                         &noise_suppression);
  }

  int agc_enabled = (auto_gain > 0) ? 1 : 0;
  speex_preprocess_ctl(state, SPEEX_PREPROCESS_SET_AGC, &agc_enabled);
  if (auto_gain > 0) {
    speex_preprocess_ctl(state, SPEEX_PREPROCESS_SET_AGC_LEVEL, &auto_gain);
  }
}

AudioProcessor::~AudioProcessor() {
  if (this->state) {
    speex_preprocess_state_destroy(this->state);
    this->state = NULL;
  }
}

std::unique_ptr<ProcessedAudioChunk>
AudioProcessor::ProcessChunk(py::bytes audio_input_py) {
  // CORRECTED LINE:
  py::buffer_info input_buf_info = py::buffer(audio_input_py).request();

  if (input_buf_info.ndim != 1) {
    throw std::runtime_error("Input audio must be a 1D bytes array.");
  }

  if (input_buf_info.size != m_chunk_size_bytes) {
    throw std::runtime_error(
        "Input audio size (" + std::to_string(input_buf_info.size) +
        " bytes) does not match configured chunk size (" +
        std::to_string(m_chunk_size_bytes) + " bytes).");
  }

  std::vector<int16_t> temp_audio_buffer(m_chunk_size_samples);
  memcpy(temp_audio_buffer.data(), input_buf_info.ptr, m_chunk_size_bytes);
  speex_preprocess_run(this->state, temp_audio_buffer.data());

  py::bytes processed_audio_py_obj(
      reinterpret_cast<const char *>(temp_audio_buffer.data()),
      m_chunk_size_bytes);

  return std::make_unique<ProcessedAudioChunk>(
      std::move(processed_audio_py_obj));
}

// ----------------------------------------------------------------------------

PYBIND11_MODULE(speex_noise_cpp, m) {
  m.doc() = R"pbdoc(
        Noise suppression using SpeexDSP
        --------------------------------
        .. currentmodule:: speex_noise_cpp
        .. autosummary::
           :toctree: _generate
           AudioProcessor
           ProcessedAudioChunk
    )pbdoc";

  py::class_<AudioProcessor>(m, "AudioProcessor")
      .def(py::init<int, float, int>(),
           py::arg("chunk_size_samples"),
           py::arg("auto_gain"),
           py::arg("noise_suppression"))
      .def("ProcessChunk", &AudioProcessor::ProcessChunk,
           py::arg("audio_input"),
           R"pbdoc(
             Processes a chunk of audio.
             The input 'audio_input' must be a bytes object containing
             'chunk_size_samples' * 2 bytes of 16-bit mono PCM audio.
             Returns a ProcessedAudioChunk object.
           )pbdoc");

  py::class_<ProcessedAudioChunk>(m, "ProcessedAudioChunk")
      .def_readonly("audio", &ProcessedAudioChunk::audio,
                    "The processed audio data as a bytes object.");

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}

