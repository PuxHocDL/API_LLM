from llama_cpp import Llama

# Kiểm tra xem CUDA có được hỗ trợ không
llm = Llama(model_path="lua_model.gguf", n_gpu_layers=1)  # Sử dụng 1 layer trên GPU
print("CUDA is supported:", llm.ctx.using_cuda)