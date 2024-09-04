#include "src/fastertransformer/devices/testing/TestBase.h"
#include "src/fastertransformer/devices/arm_impl/ArmDevice.h"
#include <torch/torch.h>
#include <chrono>

using namespace std;
using namespace fastertransformer;

class ArmLayerNormOpsTest: public DeviceTestBase {
public:
    void SetUp() override {
        DeviceTestBase::SetUp();
        rtol_ = 1e-3;
        atol_ = 1e-3;
    }

protected:
    torch::Tensor rmsNorm(const torch::Tensor& input, const torch::Tensor& gamma, const torch::Tensor& beta) {
        return input * torch::rsqrt(torch::mean(input * input, -1, true) + 1e-6) * gamma + beta;
    }

    void testGeneralLayernorm(DataType data_type, uint16_t m, uint16_t n) {
        const auto torch_dtype     = dataTypeToTorchType(data_type);
        auto       input_tensor    = (torch::arange(m * n, m * n * 2) / (n * n)).reshape({m, n}).to(torch_dtype);
        //auto       input_tensor    = (torch::arange(m * n, m * n * 2) / (n * n)).reshape({m, n}).to(torch_dtype);
        //auto       gamma_tensor    = (torch::ones({n})).to(torch_dtype);
        //auto       beta_tensor     = (torch::zeros({n})).to(torch_dtype);
        //auto       input_tensor    = torch::arange(0, m * n, 1).reshape({m, n}).to(torch_dtype);
        auto    input_tensor_copy1  = input_tensor.clone();
        auto    input_tensor_copy2  = input_tensor.clone();

        auto       gamma_tensor    = (torch::rand({n})).to(torch_dtype);
        auto       beta_tensor     = (torch::rand({n})).to(torch_dtype);
        //auto       residual_tensor = torch::arange(0, m * n, 1).reshape({m, n}).to(torch_dtype);
        auto       residual_tensor = torch::arange(0, m * n, 1).reshape({m, n}).to(torch_dtype);
        auto       bias_tesnsor = (torch::rand({n})).to(torch_dtype);
        auto       bias_tensor1 = (torch::empty({m, n})).to(torch_dtype);
        for (int i = 0; i < m; ++i) {
            bias_tensor1[i] = bias_tesnsor;
        }

        auto      bias    = tensorToBuffer(bias_tesnsor, AllocationType::HOST);
        //auto      residual_tensor    = (torch::rand({m, n})).to(torch_dtype);
        auto      input   = tensorToBuffer(input_tensor, AllocationType::HOST);
        auto      gamma   = tensorToBuffer(gamma_tensor, AllocationType::HOST);
        auto      beta    = tensorToBuffer(beta_tensor, AllocationType::HOST);
        auto      weights = LayerNormWeights(gamma, beta);
        BufferPtr empty;
        auto      gamma_only_weights = LayerNormWeights(gamma, empty);
        auto      residual           = tensorToBuffer(residual_tensor, AllocationType::HOST);
        auto      plus_tensor = input_tensor + residual_tensor + bias_tensor1;


        std::cout<<"datatype:"<<(int)data_type<<std::endl;
        std::cout<<"n:"<< n <<std::endl;

        // test case 1: rmsnorm without residual/bias
        auto testcase1_output = device_->layernorm(LayernormParams(input,
                                                                   nullptr,
                                                                   weights,
                                                                   std::nullopt,
                                                                   std::nullopt,
                                                                   std::nullopt,
                                                                   0.f,
                                                                   1e-6,
                                                                   false,
                                                                   false,
                                                                   NormType::rmsnorm));
        // auto expected_output1 = torch::layer_norm(//plus_tensor.to(torch::kFloat32),
        //                                          input_tensor_copy1.to(torch::kFloat32),
        //                                          {n},
        //                                          gamma_tensor.to(torch::kFloat32),
        //                                          beta_tensor.to(torch::kFloat32),
        //                                          1e-6);
        auto expected_output1 = rmsNorm(
            input_tensor.to(torch::kFloat32), gamma_tensor.to(torch::kFloat32), beta_tensor.to(torch::kFloat32));
        assertTensorClose(expected_output1, bufferToTensor(*(testcase1_output.output)));        
        
        // test case 2: rmsnorm with residual/bias
        auto testcase2_output = device_->layernorm(LayernormParams(input,
                                                                   nullptr,
                                                                   weights,
                                                                   std::cref(*residual),
                                                                   std::nullopt,
                                                                   std::cref(*bias),
                                                                   0.f,
                                                                   1e-6,
                                                                   false,
                                                                   false,
                                                                   NormType::rmsnorm));
        // auto expected_output2 = torch::layer_norm(plus_tensor2.to(torch::kFloat32),
        //                                          {n},
        //                                          gamma_tensor.to(torch::kFloat32),
        //                                          beta_tensor.to(torch::kFloat32),
        //                                          1e-6);
        auto expected_output2 = rmsNorm(
            plus_tensor.to(torch::kFloat32), gamma_tensor.to(torch::kFloat32), beta_tensor.to(torch::kFloat32));        
        assertTensorClose(expected_output2, bufferToTensor(*(testcase2_output .output)));  

        // test case 3: layernorm without residual/bias
        auto testcase3_output = device_->layernorm(LayernormParams(input,
                                                                   nullptr,
                                                                   weights,
                                                                   std::nullopt,
                                                                   std::nullopt,
                                                                   std::nullopt,
                                                                   0.f,
                                                                   1e-6,
                                                                   false,
                                                                   false,
                                                                   NormType::layernorm));
        auto expected_output3 = torch::layer_norm(//plus_tensor.to(torch::kFloat32),
                                                 input_tensor_copy1.to(torch::kFloat32),
                                                 {n},
                                                 gamma_tensor.to(torch::kFloat32),
                                                 beta_tensor.to(torch::kFloat32),
                                                 1e-6);

        assertTensorClose(expected_output3, bufferToTensor(*(testcase3_output.output)));        
        
        // test case 4: layernorm with residual/bias
        auto testcase4_output = device_->layernorm(LayernormParams(input,
                                                                   nullptr,
                                                                   weights,
                                                                   std::cref(*residual),
                                                                   std::nullopt,
                                                                   std::cref(*bias),
                                                                   0.f,
                                                                   1e-6,
                                                                   false,
                                                                   false,
                                                                   NormType::layernorm));
        auto expected_output4 = torch::layer_norm(plus_tensor.to(torch::kFloat32),
                                                 {n},
                                                 gamma_tensor.to(torch::kFloat32),
                                                 beta_tensor.to(torch::kFloat32),
                                                 1e-6);
  
        assertTensorClose(expected_output4, bufferToTensor(*(testcase4_output .output))); 

        // std::chrono::duration<double> duration;
        // for(int i = 0;i<100;i++){
        //     input_tensor    = (torch::rand({m, n})).to(torch_dtype);
        //     gamma_tensor    = (torch::rand({n})).to(torch_dtype);
        //     beta_tensor     = (torch::rand({n})).to(torch_dtype);
        //     residual_tensor = (torch::rand({m, n})).to(torch_dtype);

        //     input   = tensorToBuffer(input_tensor, AllocationType::HOST);
        //     gamma   = tensorToBuffer(gamma_tensor, AllocationType::HOST);
        //     beta    = tensorToBuffer(beta_tensor, AllocationType::HOST);
        //     weights = LayerNormWeights(gamma, beta);
        //                         device_->layernorm(LayernormParams(input,
        //                                                            nullptr,
        //                                                            weights,
        //                                                            std::nullopt,
        //                                                            std::nullopt,
        //                                                            std::nullopt,
        //                                                            0.f,
        //                                                            1e-6,
        //                                                            false,
        //                                                            false,
        //                                                            NormType::layernorm));
        // }         
        // for(int i = 0;i<100;i++){
        //     input_tensor    = (torch::rand({m, n})).to(torch_dtype);
        //     gamma_tensor    = (torch::rand({n})).to(torch_dtype);
        //     beta_tensor     = (torch::rand({n})).to(torch_dtype);
        //     residual_tensor = (torch::rand({m, n})).to(torch_dtype);

        //     input   = tensorToBuffer(input_tensor, AllocationType::HOST);
        //     gamma   = tensorToBuffer(gamma_tensor, AllocationType::HOST);
        //     beta    = tensorToBuffer(beta_tensor, AllocationType::HOST);
        //     weights = LayerNormWeights(gamma, beta);
            
            
        //     //auto start = std::chrono::high_resolution_clock::now();
        //                         device_->layernorm(LayernormParams(input,
        //                                                            nullptr,
        //                                                            weights,
        //                                                            std::nullopt,
        //                                                            std::nullopt,
        //                                                            std::nullopt,
        //                                                            0.f,
        //                                                            1e-6,
        //                                                            false,
        //                                                            false,
        //                                                            NormType::layernorm));
        //     //auto end = std::chrono::high_resolution_clock::now();
        //     //duration += (end - start); 
        // }                                                                 
        //std::cout << "运行时间: " << duration.count() << " 秒" << std::endl;

        // test case 2: rms norm without residual
        // auto testcase2_output = device_->layernorm(LayernormParams(input,
        //                                                            nullptr,
        //                                                            weights,
        //                                                            std::nullopt,
        //                                                            std::nullopt,
        //                                                            std::nullopt,
        //                                                            0.f,
        //                                                            1e-6,
        //                                                            false,
        //                                                            false,
        //                                                            NormType::rmsnorm));
        // //torch::kHalf
        // auto expected_output = rmsNorm(
        //     input_tensor.to(torch::kFloat32), gamma_tensor.to(torch::kFloat32), beta_tensor.to(torch::kFloat32));
        // assertTensorClose(expected_output, bufferToTensor(*testcase2_output.output));

        // std::chrono::duration<double> duration;
        // for(int i = 0;i<1000;i++){
        //     input_tensor    = (torch::rand({m, n})).to(torch_dtype);
        //     gamma_tensor    = (torch::rand({n})).to(torch_dtype);
        //     beta_tensor     = (torch::rand({n})).to(torch_dtype);
        //     residual_tensor = (torch::rand({m, n})).to(torch_dtype);

        //     input   = tensorToBuffer(input_tensor, AllocationType::HOST);
        //     gamma   = tensorToBuffer(gamma_tensor, AllocationType::HOST);
        //     beta    = tensorToBuffer(beta_tensor, AllocationType::HOST);
        //     weights = LayerNormWeights(gamma, beta);
        //                         device_->layernorm(LayernormParams(input,
        //                                                            nullptr,
        //                                                            weights,
        //                                                            std::nullopt,
        //                                                            std::nullopt,
        //                                                            std::nullopt,
        //                                                            0.f,
        //                                                            1e-6,
        //                                                            false,
        //                                                            false,
        //                                                            NormType::rmsnorm));
        // }  
        // for(int i = 0;i<10000;i++){
        //     input_tensor    = (torch::rand({m, n})).to(torch_dtype);
        //     gamma_tensor    = (torch::rand({n})).to(torch_dtype);
        //     beta_tensor     = (torch::rand({n})).to(torch_dtype);
        //     residual_tensor = (torch::rand({m, n})).to(torch_dtype);

        //     input   = tensorToBuffer(input_tensor, AllocationType::HOST);
        //     gamma   = tensorToBuffer(gamma_tensor, AllocationType::HOST);
        //     beta    = tensorToBuffer(beta_tensor, AllocationType::HOST);
        //     weights = LayerNormWeights(gamma, beta);
            
            
        //     //auto start = std::chrono::high_resolution_clock::now();
        //                         device_->layernorm(LayernormParams(input,
        //                                                            nullptr,
        //                                                            weights,
        //                                                            std::nullopt,
        //                                                            std::nullopt,
        //                                                            std::nullopt,
        //                                                            0.f,
        //                                                            1e-6,
        //                                                            false,
        //                                                            false,
        //                                                            NormType::rmsnorm));

        //     //auto end = std::chrono::high_resolution_clock::now();
        //     //duration += (end - start); 
        // }                                                                 
        //std::cout << "运行时间: " << duration.count() << " 秒" << std::endl;

    }
};

TEST_F(ArmLayerNormOpsTest, testSimpleLayernorm) {
    const auto test_m = vector<uint16_t>({8});
    const auto test_n = vector<uint16_t>({2048});
    for (const auto& m : test_m) {
        for (const auto& n : test_n) {
            testGeneralLayernorm(DataType::TYPE_FP32, m, n);
            //testGeneralLayernorm(DataType::TYPE_FP16, m, n);
        }
    }
}
