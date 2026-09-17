#include "gewell/models/gemma4/31b/artifact.h"
#include "artifact_detail.h"

#include <openssl/evp.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

namespace gewell::artifact {
namespace {

using namespace detail;

namespace model = gemma4_31b;

constexpr std::size_t kHeaderUsedBytes = 328;
constexpr std::size_t kHeaderHashOffset = 296;
constexpr std::uint32_t kScalarTypeBf16Le = 1;
constexpr std::uint32_t kLayoutCOrder = 1;
constexpr std::uint32_t kTargetSm120a = 1;


static_assert(static_cast<std::uint16_t>(model::TensorRole::embedding) == 0);
static_assert(static_cast<std::uint16_t>(model::TensorRole::assistant_embedding) == 34);
static_assert(static_cast<std::uint16_t>(model::TensorRole::assistant_input_norm) == 35);
static_assert(static_cast<std::uint16_t>(model::TensorRole::assistant_q_proj) == 36);
static_assert(static_cast<std::uint16_t>(model::TensorRole::assistant_q_norm) == 37);
static_assert(static_cast<std::uint16_t>(model::TensorRole::assistant_o_proj) == 38);
static_assert(static_cast<std::uint16_t>(model::TensorRole::assistant_post_attention_norm) == 39);
static_assert(static_cast<std::uint16_t>(model::TensorRole::assistant_pre_feedforward_norm) == 40);
static_assert(static_cast<std::uint16_t>(model::TensorRole::assistant_gate_proj) == 41);
static_assert(static_cast<std::uint16_t>(model::TensorRole::assistant_up_proj) == 42);
static_assert(static_cast<std::uint16_t>(model::TensorRole::assistant_down_proj) == 43);
static_assert(static_cast<std::uint16_t>(model::TensorRole::assistant_post_feedforward_norm) == 44);
static_assert(static_cast<std::uint16_t>(model::TensorRole::assistant_layer_scalar) == 45);
static_assert(static_cast<std::uint16_t>(model::TensorRole::assistant_final_norm) == 46);
static_assert(static_cast<std::uint16_t>(model::TensorRole::assistant_pre_projection) == 47);
static_assert(static_cast<std::uint16_t>(model::TensorRole::assistant_post_projection) == 48);

static_assert(static_cast<std::uint8_t>(model::DType::bf16) ==
              kScalarTypeBf16Le);
static_assert(static_cast<std::uint16_t>(model::TensorRole::input_norm) == 1);
static_assert(static_cast<std::uint16_t>(model::TensorRole::q_proj) == 2);
static_assert(static_cast<std::uint16_t>(model::TensorRole::k_proj) == 3);
static_assert(static_cast<std::uint16_t>(model::TensorRole::v_proj) == 4);
static_assert(static_cast<std::uint16_t>(model::TensorRole::q_norm) == 5);
static_assert(static_cast<std::uint16_t>(model::TensorRole::k_norm) == 6);
static_assert(static_cast<std::uint16_t>(model::TensorRole::o_proj) == 7);
static_assert(
    static_cast<std::uint16_t>(model::TensorRole::post_attention_norm) == 8);
static_assert(
    static_cast<std::uint16_t>(model::TensorRole::pre_feedforward_norm) == 9);
static_assert(static_cast<std::uint16_t>(model::TensorRole::gate_proj) == 10);
static_assert(static_cast<std::uint16_t>(model::TensorRole::up_proj) == 11);
static_assert(static_cast<std::uint16_t>(model::TensorRole::down_proj) == 12);
static_assert(
    static_cast<std::uint16_t>(model::TensorRole::post_feedforward_norm) == 13);
static_assert(static_cast<std::uint16_t>(model::TensorRole::layer_scalar) == 14);
static_assert(static_cast<std::uint16_t>(model::TensorRole::final_norm) == 15);
static_assert(
    static_cast<std::uint16_t>(model::TensorRole::vision_patch_proj) == 16);
static_assert(static_cast<std::uint16_t>(
                  model::TensorRole::vision_position_embedding) == 17);
static_assert(
    static_cast<std::uint16_t>(model::TensorRole::vision_input_norm) == 18);
static_assert(static_cast<std::uint16_t>(model::TensorRole::vision_q_proj) ==
              19);
static_assert(static_cast<std::uint16_t>(model::TensorRole::vision_k_proj) ==
              20);
static_assert(static_cast<std::uint16_t>(model::TensorRole::vision_v_proj) ==
              21);
static_assert(static_cast<std::uint16_t>(model::TensorRole::vision_q_norm) ==
              22);
static_assert(static_cast<std::uint16_t>(model::TensorRole::vision_k_norm) ==
              23);
static_assert(static_cast<std::uint16_t>(model::TensorRole::vision_o_proj) ==
              24);
static_assert(static_cast<std::uint16_t>(
                  model::TensorRole::vision_post_attention_norm) == 25);
static_assert(static_cast<std::uint16_t>(
                  model::TensorRole::vision_pre_feedforward_norm) == 26);
static_assert(
    static_cast<std::uint16_t>(model::TensorRole::vision_gate_proj) == 27);
static_assert(static_cast<std::uint16_t>(model::TensorRole::vision_up_proj) ==
              28);
static_assert(
    static_cast<std::uint16_t>(model::TensorRole::vision_down_proj) == 29);
static_assert(static_cast<std::uint16_t>(
                  model::TensorRole::vision_post_feedforward_norm) == 30);
static_assert(static_cast<std::uint16_t>(model::TensorRole::vision_std_bias) ==
              31);
static_assert(static_cast<std::uint16_t>(model::TensorRole::vision_std_scale) ==
              32);
static_assert(
    static_cast<std::uint16_t>(model::TensorRole::vision_projection) == 33);
static_assert(kDataOffset ==
              model::align_up(kHeaderBytes +
                                  model::kTextPhysicalTensorCount * kEntryBytes,
                              model::kStorageAlignment));
static_assert(kLogicalDataBytes == model::logical_text_weight_bytes());
static_assert(kPayloadBytes == model::aligned_text_weight_bytes());
static_assert(kFileBytes == kDataOffset + kPayloadBytes);

std::string system_error(const std::string& operation,
                         const std::string& path) {
  return operation + " " + path + ": " + std::strerror(errno);
}

class EvpSha256 {
 public:
  EvpSha256() : context_(EVP_MD_CTX_new()) {
    require(context_ != nullptr, "OpenSSL SHA-256 context allocation failed");
    if (EVP_DigestInit_ex(context_, EVP_sha256(), nullptr) != 1) {
      EVP_MD_CTX_free(context_);
      context_ = nullptr;
      fail("OpenSSL SHA-256 initialization failed");
    }
  }

  ~EvpSha256() { EVP_MD_CTX_free(context_); }
  EvpSha256(const EvpSha256&) = delete;
  EvpSha256& operator=(const EvpSha256&) = delete;

  void Update(const std::uint8_t* bytes, std::size_t length) {
    require(EVP_DigestUpdate(context_, bytes, length) == 1,
            "OpenSSL SHA-256 update failed");
  }

  [[nodiscard]] Digest Finish() {
    Digest digest{};
    unsigned length = 0;
    require(EVP_DigestFinal_ex(context_, digest.data(), &length) == 1,
            "OpenSSL SHA-256 finalization failed");
    require(length == digest.size(), "OpenSSL returned a wrong SHA-256 length");
    return digest;
  }

 private:
  EVP_MD_CTX* context_{};
};

Digest read_digest(const std::uint8_t* bytes) {
  Digest result{};
  std::copy_n(bytes, result.size(), result.begin());
  return result;
}

bool all_zero(const std::uint8_t* bytes, std::size_t length) {
  for (std::size_t index = 0; index < length; ++index) {
    if (bytes[index] != 0) {
      return false;
    }
  }
  return true;
}

std::uint64_t checked_add(std::uint64_t first, std::uint64_t second,
                          const std::string& label) {
  require(second <= std::numeric_limits<std::uint64_t>::max() - first,
          label + " overflows u64");
  return first + second;
}

std::string tensor_label(std::size_t id, std::string_view detail) {
  return "tensor " + std::to_string(id) + " has wrong " +
         std::string(detail);
}

}  // namespace

std::string digest_hex(const Digest& digest) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string result(digest.size() * 2, '0');
  for (std::size_t index = 0; index < digest.size(); ++index) {
    result[index * 2] = digits[digest[index] >> 4];
    result[index * 2 + 1] = digits[digest[index] & 0x0f];
  }
  return result;
}

ArtifactFile::ArtifactFile(std::string path, int descriptor,
                           const std::uint8_t* mapping,
                           std::size_t mapping_bytes)
    : path_(std::move(path)),
      descriptor_(descriptor),
      mapping_(mapping),
      mapping_bytes_(mapping_bytes) {}

ArtifactFile::~ArtifactFile() { Reset(); }

ArtifactFile::ArtifactFile(ArtifactFile&& other) noexcept
    : path_(std::move(other.path_)),
      descriptor_(std::exchange(other.descriptor_, -1)),
      mapping_(std::exchange(other.mapping_, nullptr)),
      mapping_bytes_(std::exchange(other.mapping_bytes_, 0)),
      header_(other.header_),
      entries_(other.entries_) {}

ArtifactFile& ArtifactFile::operator=(ArtifactFile&& other) noexcept {
  if (this != &other) {
    Reset();
    path_ = std::move(other.path_);
    descriptor_ = std::exchange(other.descriptor_, -1);
    mapping_ = std::exchange(other.mapping_, nullptr);
    mapping_bytes_ = std::exchange(other.mapping_bytes_, 0);
    header_ = other.header_;
    entries_ = other.entries_;
  }
  return *this;
}

void ArtifactFile::Reset() noexcept {
  if (mapping_ != nullptr) {
    ::munmap(const_cast<std::uint8_t*>(mapping_), mapping_bytes_);
    mapping_ = nullptr;
    mapping_bytes_ = 0;
  }
  if (descriptor_ >= 0) {
    ::close(descriptor_);
    descriptor_ = -1;
  }
}

ArtifactFile ArtifactFile::Open(const std::string& path) {
  const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (descriptor < 0) {
    fail(system_error("cannot open artifact", path));
  }

  struct stat status {};
  if (::fstat(descriptor, &status) != 0) {
    const std::string message = system_error("cannot stat artifact", path);
    ::close(descriptor);
    fail(message);
  }
  if (!S_ISREG(status.st_mode)) {
    ::close(descriptor);
    fail("artifact is not a regular file: " + path);
  }
  if (status.st_size < static_cast<off_t>(kDataOffset) ||
      static_cast<std::uint64_t>(status.st_size) > kFileBytes) {
    const std::uint64_t actual =
        status.st_size < 0 ? 0 : static_cast<std::uint64_t>(status.st_size);
    ::close(descriptor);
    fail("artifact file size mismatch: got " + std::to_string(actual) +
         ", outside supported artifact bounds");
  }
  require(kFileBytes <= std::numeric_limits<std::size_t>::max(),
          "artifact does not fit this process address space");
  const std::size_t mapping_bytes = static_cast<std::size_t>(status.st_size);
  void* mapping =
      ::mmap(nullptr, mapping_bytes, PROT_READ, MAP_PRIVATE, descriptor, 0);
  if (mapping == MAP_FAILED) {
    const std::string message = system_error("cannot mmap artifact", path);
    ::close(descriptor);
    fail(message);
  }

  ArtifactFile result(path, descriptor,
                      static_cast<const std::uint8_t*>(mapping), mapping_bytes);
  result.ValidateMetadata();
  return result;
}

void ArtifactFile::ValidateMetadata() {
  require(mapping_ != nullptr && mapping_bytes_ >= kDataOffset,
          "artifact mapping is invalid");
  const std::uint8_t* raw = mapping_;
  static constexpr std::array<std::uint8_t, 8> magic{{
      'G', 'E', 'W', 'B', 'F', '1', '6', 0,
  }};
  static constexpr std::array<std::uint8_t, 8> nvfp4_magic{{
      'G', 'E', 'W', 'N', 'V', 'F', '4', 0,
  }};
  static constexpr std::array<std::uint8_t, 8> mixed_magic{{
      'G', 'E', 'W', 'M', 'I', 'X', '1', 0,
  }};
  header_.native_nvfp4 = std::equal(nvfp4_magic.begin(), nvfp4_magic.end(), raw);
  header_.native_mixed = std::equal(mixed_magic.begin(), mixed_magic.end(), raw);
  const bool native_profile = header_.native_nvfp4 || header_.native_mixed;
  require(native_profile || std::equal(magic.begin(), magic.end(), raw), "wrong magic");
  header_.format_version = read_u32_le(raw + 8);
  require(header_.format_version == (header_.native_mixed ? kMixedFormatVersion :
                                     header_.native_nvfp4 ? kNvfp4FormatVersion : kFormatVersion),
          "wrong format version");
  require(read_u32_le(raw + 12) == kHeaderBytes, "wrong header bytes");
  require(read_u32_le(raw + 16) == kEntryBytes, "wrong entry bytes");

  header_.physical_tensor_count = read_u32_le(raw + 20);
  header_.logical_tensor_count = read_u32_le(raw + 24);
  header_.alias_count = read_u32_le(raw + 28);
  require(header_.physical_tensor_count == model::kTextPhysicalTensorCount,
          "wrong physical tensor count");
  require(header_.logical_tensor_count == model::kTextPhysicalTensorCount + 1,
          "wrong logical tensor count");
  require(header_.alias_count == 1, "wrong alias count");
  require(read_u32_le(raw + 32) == model::kStorageAlignment,
          "wrong alignment");
  require(read_u32_le(raw + 36) == kScalarTypeBf16Le, "wrong scalar type");
  require(read_u32_le(raw + 40) == kLayoutCOrder, "wrong layout");
  require(read_u32_le(raw + 44) == kTargetSm120a, "wrong target");
  require(read_u64_le(raw + 48) == kEntriesOffset, "wrong entries offset");

  header_.data_offset = read_u64_le(raw + 56);
  header_.logical_data_bytes = read_u64_le(raw + 64);
  header_.payload_bytes = read_u64_le(raw + 72);
  header_.file_bytes = read_u64_le(raw + 80);
  header_.lm_head_logical_id = read_u32_le(raw + 88);
  header_.lm_head_target_id = read_u32_le(raw + 92);
  require(header_.data_offset == kDataOffset, "wrong data offset");
  if (!native_profile) {
    require(header_.logical_data_bytes == kLogicalDataBytes, "wrong logical data byte count");
    require(header_.payload_bytes == kPayloadBytes, "wrong payload byte count");
    require(header_.file_bytes == kFileBytes, "wrong declared file size");
  }
  require(header_.file_bytes == mapping_bytes_, "artifact file size mismatch");
  require(header_.lm_head_logical_id == model::kLmHeadLogicalId,
          "wrong lm_head logical id");
  require(header_.lm_head_target_id == model::kEmbeddingPhysicalId,
          "wrong lm_head target id");
  header_.config_sha256 = read_digest(raw + 168);
  header_.source_index_sha256 = read_digest(raw + 200);
  header_.entry_table_sha256 = read_digest(raw + 232);
  header_.payload_sha256 = read_digest(raw + 264);
  header_.header_sha256 = read_digest(raw + 296);
  require(all_zero(raw + kHeaderUsedBytes, kHeaderBytes - kHeaderUsedBytes),
          "header reserved bytes are nonzero");

  Sha256 header_digest;
  header_digest.Update(raw, kHeaderHashOffset);
  const std::array<std::uint8_t, 32> zeros{};
  header_digest.Update(zeros.data(), zeros.size());
  header_digest.Update(raw + kHeaderHashOffset + zeros.size(),
                       kHeaderBytes - kHeaderHashOffset - zeros.size());
  require(header_digest.Finish() == header_.header_sha256,
          "header SHA-256 mismatch");

  const std::uint64_t table_bytes =
      model::kTextPhysicalTensorCount * kEntryBytes;
  require(checked_add(kEntriesOffset, table_bytes, "tensor table") <=
              header_.data_offset,
          "data offset precedes tensor table");
  const std::uint8_t* table = raw + kEntriesOffset;
  require(hash_bytes(table, static_cast<std::size_t>(table_bytes)) ==
              header_.entry_table_sha256,
          "entry-table SHA-256 mismatch");
  const std::uint64_t index_padding =
      header_.data_offset - kEntriesOffset - table_bytes;
  require(all_zero(table + table_bytes, static_cast<std::size_t>(index_padding)),
          "index padding is nonzero");

  std::uint64_t cursor = header_.data_offset;
  std::uint64_t logical_bytes = 0;
  std::uint64_t payload_bytes = 0;
  for (std::size_t index = 0; index < entries_.size(); ++index) {
    const std::uint8_t* encoded = table + index * kEntryBytes;
    TensorEntry entry{};
    entry.physical_id = read_u16_le(encoded);
    entry.layer = read_i16_le(encoded + 2);
    const std::uint16_t role = read_u16_le(encoded + 4);
    require(role <=
                static_cast<std::uint16_t>(model::TensorRole::assistant_post_projection),
            "entry " + std::to_string(index) + " is malformed");
    entry.role = static_cast<model::TensorRole>(role);
    entry.rank = encoded[6];
    require(encoded[7] <= (header_.native_mixed ? 2 : header_.native_nvfp4 ? 1 : 0),
            "entry " + std::to_string(index) + " has unsupported storage type");
    entry.storage_type = static_cast<StorageType>(encoded[7]);
    entry.dim0 = read_u32_le(encoded + 8);
    entry.dim1 = read_u32_le(encoded + 12);
    entry.dim2 = read_u32_le(encoded + 16);
    entry.file_offset = read_u64_le(encoded + 20);
    entry.byte_length = read_u64_le(encoded + 28);
    entry.sha256 = read_digest(encoded + 36);
    require(all_zero(encoded + 68, 4),
            "entry " + std::to_string(index) +
                " reserved tail is nonzero");

    const model::TensorSpec& spec = model::kPhysicalTensors[index];
    require(entry.physical_id == index, tensor_label(index, "id"));
    require(entry.layer == spec.layer, tensor_label(index, "layer"));
    require(entry.role == spec.role, tensor_label(index, "role"));
    require(entry.rank == spec.shape.rank, tensor_label(index, "rank"));
    require(entry.dim0 == spec.shape.dimensions[0],
            tensor_label(index, "dim0"));
    require(entry.dim1 == spec.shape.dimensions[1],
            tensor_label(index, "dim1"));
    require(entry.dim2 == spec.shape.dimensions[2],
            tensor_label(index, "dim2"));
    require(entry.file_offset == cursor, tensor_label(index, "offset"));
    const bool nvfp4 = entry.storage_type == StorageType::nvfp4_w4a4;
    const bool fp8 = entry.storage_type == StorageType::fp8_w8a8;
    const bool native = nvfp4 || fp8;
    const bool mlp_role = entry.role == model::TensorRole::gate_proj ||
                        entry.role == model::TensorRole::up_proj ||
                        entry.role == model::TensorRole::down_proj;
    require(!native || mlp_role || entry.role == model::TensorRole::q_proj ||
                entry.role == model::TensorRole::k_proj || entry.role == model::TensorRole::v_proj ||
                entry.role == model::TensorRole::o_proj, tensor_label(index, "native projection role"));
    require(!native || (entry.layer >= 0 && entry.rank == 2 && entry.dim0 > 0 &&
                        entry.dim1 > 0 && (!nvfp4 || entry.dim1 % 16 == 0)),
            tensor_label(index, "packed dimensions"));
    const auto packed_bytes = nvfp4 ? nvfp4_packed_bytes(entry.dim0, entry.dim1) :
                              fp8 ? fp8_packed_bytes(entry.dim0, entry.dim1) : 0;
    const auto scale_bytes = nvfp4 ? nvfp4_scale_bytes(entry.dim0, entry.dim1) : 0;
    require(entry.byte_length == (native ? packed_bytes + scale_bytes + 8 : spec.byte_count()),
            tensor_label(index, "byte length"));

    const std::uint64_t slot_bytes = checked_align_up(
        entry.byte_length, model::kStorageAlignment,
        "tensor " + std::to_string(index));
    cursor = checked_add(cursor, slot_bytes,
                         "tensor " + std::to_string(index) + " end");
    logical_bytes = checked_add(logical_bytes, entry.byte_length,
                                "logical data byte count");
    payload_bytes =
        checked_add(payload_bytes, slot_bytes, "payload byte count");
    require(cursor <= header_.file_bytes,
            "tensor " + std::to_string(index) + " exceeds artifact");
    if (native) {
      const auto* globals = mapping_ + entry.file_offset + packed_bytes + scale_bytes;
      const auto weight_bits = read_u32_le(globals);
      const auto input_bits = read_u32_le(globals + 4);
      std::memcpy(&entry.weight_scale_2, &weight_bits, sizeof(float));
      std::memcpy(&entry.input_scale, &input_bits, sizeof(float));
      require(std::isfinite(entry.weight_scale_2) && entry.weight_scale_2 > 0 &&
                  std::isfinite(entry.input_scale) && entry.input_scale > 0,
              "tensor " + std::to_string(index) + " has invalid packed global scales");
      if (fp8) {
        const float input_inverse = 1.0f / entry.input_scale;
        const float output_scale = entry.weight_scale_2 * entry.input_scale;
        require(std::isfinite(input_inverse) && std::isfinite(output_scale) && output_scale > 0,
                "tensor " + std::to_string(index) + " has FP8 global scales outside the executable FP32 range");
      }
    }
    entries_[index] = entry;
  }

  require(header_.logical_data_bytes == logical_bytes,
          "logical data byte count mismatch");
  require(header_.payload_bytes == payload_bytes,
          "payload byte count mismatch");
  require(checked_add(header_.data_offset, payload_bytes, "declared file size") ==
              header_.file_bytes,
          "declared file size is inconsistent");
  require(cursor == header_.file_bytes,
          "tensor table does not cover the payload");
}

const std::uint8_t* ArtifactFile::payload_data() const {
  require(mapping_ != nullptr, "artifact is not open");
  return mapping_ + header_.data_offset;
}

const std::uint8_t* ArtifactFile::tensor_data(std::size_t physical_id) const {
  require(mapping_ != nullptr, "artifact is not open");
  require(physical_id < entries_.size(), "physical tensor id is out of range");
  return mapping_ + entries_[physical_id].file_offset;
}

Verification ArtifactFile::VerifyFull() const {
  require(mapping_ != nullptr, "artifact is not open");
  ::madvise(const_cast<std::uint8_t*>(mapping_), mapping_bytes_, MADV_SEQUENTIAL);

  EvpSha256 payload_digest;

  for (std::size_t index = 0; index < entries_.size(); ++index) {
    const TensorEntry& entry = entries_[index];
    const std::uint8_t* cursor = mapping_ + entry.file_offset;
    EvpSha256 tensor_digest;
    std::uint64_t remaining = entry.byte_length;
    while (remaining != 0) {
      const std::size_t chunk = static_cast<std::size_t>(
          std::min<std::uint64_t>(remaining, kIoChunkBytes));
      payload_digest.Update(cursor, chunk);
      tensor_digest.Update(cursor, chunk);
      cursor += chunk;
      remaining -= chunk;
    }
    require(tensor_digest.Finish() == entry.sha256,
            "tensor " + std::to_string(index) + " SHA-256 mismatch");
    if (entry.storage_type == StorageType::nvfp4_w4a4) {
      const auto* scales = mapping_ + entry.file_offset +
                           nvfp4_packed_bytes(entry.dim0, entry.dim1);
      const auto scales_length = nvfp4_scale_bytes(entry.dim0, entry.dim1);
      require(std::all_of(scales, scales + scales_length,
                          [](std::uint8_t value) { return value < 0x7f; }),
              "tensor " + std::to_string(index) + " has invalid NVFP4 block scales");
    }
    if (entry.storage_type == StorageType::fp8_w8a8) {
      const auto* weights = mapping_ + entry.file_offset;
      require(std::all_of(weights, weights + fp8_packed_bytes(entry.dim0, entry.dim1),
                          [](std::uint8_t value) { return (value & 0x7f) != 0x7f; }),
              "tensor " + std::to_string(index) + " has invalid FP8 weights");
    }

    std::uint64_t padding =
        checked_align_up(entry.byte_length, model::kStorageAlignment,
                         "tensor " + std::to_string(index)) -
        entry.byte_length;
    while (padding != 0) {
      const std::size_t chunk = static_cast<std::size_t>(
          std::min<std::uint64_t>(padding, kIoChunkBytes));
      require(all_zero(cursor, chunk),
              "padding after tensor " + std::to_string(index) +
                  " is nonzero");
      payload_digest.Update(cursor, chunk);
      cursor += chunk;
      padding -= chunk;
    }
  }

  const Digest actual_payload_sha256 = payload_digest.Finish();
  require(actual_payload_sha256 == header_.payload_sha256,
          "payload SHA-256 mismatch");
  Verification result{};
  result.payload_sha256 = actual_payload_sha256;
  return result;
}

}  // namespace gewell::artifact
