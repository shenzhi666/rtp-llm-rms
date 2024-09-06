#include "src/fastertransformer/devices/arm_impl/ArmDevice.h"
#include <thread>
namespace fastertransformer {

BufferPtr ArmCpuDevice::embeddingLookup(const EmbeddingLookupParams& params) {
    const auto& tokens          = params.combo_tokens;
    const auto& embedding_table = params.embedding_table;
    const auto& position_ids    = params.position_ids;
    const auto& position_table  = params.position_table;

    const auto token_num      = tokens.size();
    const auto hidden_size    = embedding_table.shape()[1];
    const auto data_type      = embedding_table.type();
    const auto data_type_size = getTypeSize(data_type);

    auto embeddings = allocateBuffer({data_type, {token_num, hidden_size}, AllocationType::HOST});

    int copy_size = hidden_size * data_type_size;

    // select the rows from embedding table
    #pragma omp parallel for num_threads(std::min((int)(token_num/2),(int)std::thread::hardware_concurrency())) if(token_num>2)
    for (int index = 0; index < token_num; index++) {
        int row_index  = tokens.data<int>()[index];
        int src_offset = row_index * copy_size;
        int dst_offset = index * copy_size;

        std::memcpy(embeddings->data() + dst_offset, embedding_table.data() + src_offset, copy_size);
    }

    return move(embeddings);
}

}  // namespace fastertransformer
