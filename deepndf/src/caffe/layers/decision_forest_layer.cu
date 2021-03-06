#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/decision_forest_layer.hpp"
#include "caffe/util/math_functions.hpp"

#define sigmoid(x) (1. / (1. + exp(-x)))
#define eps 1e-8

namespace caffe {

template <typename Dtype>
__global__ void DecisionForestGetD(const int nthreads, 
                     Dtype* d_data, const Dtype* bottom_data){
  CUDA_KERNEL_LOOP(index, nthreads) {
    d_data[index] = sigmoid(bottom_data[index]);
  }
}

template <typename Dtype>
__global__ void DecisionForestGetDP(const int nthreads,
          Dtype* d_data, Dtype* dp_data, const int high_bit,
		  const int N, const int num_output,
          const int depth, const int tree_num) {
  
  const int leaf_num_per_tree = 1 << (depth - 1);
  const int node_num_per_tree = leaf_num_per_tree - 1;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / (tree_num * node_num_per_tree);
    const int t = index / node_num_per_tree % tree_num;
    const int l = index % node_num_per_tree;

    if (l == 0) {
      dp_data[index] = (Dtype)1.;
    } else if ((high_bit << 1 ) > l + 1 && ((l + 1) & high_bit)) {
      if (l%2) {
        dp_data[index] = d_data[(n * tree_num + t) * node_num_per_tree + (l >> 1)] 
                    * dp_data[(n * tree_num + t) * node_num_per_tree + (l >> 1)];
      } else {
        dp_data[index] = (1 - d_data[(n * tree_num + t) * node_num_per_tree + (l >> 1) - 1])
                    * dp_data[(n * tree_num + t) * node_num_per_tree + (l >> 1) - 1];
      }
    }
  }
}

template <typename Dtype>
__global__ void DecisionForestGetMu(const int nthreads,
          Dtype* d_data, Dtype* mu_data, Dtype* dp_data, const int N, 
          const int num_output, const int depth, const int tree_num) {

  const int leaf_num_per_tree = 1 << (depth - 1);
  const int node_num_per_tree = leaf_num_per_tree - 1;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int tree_idx = index / leaf_num_per_tree;
    const int leaf_idx = index % leaf_num_per_tree;
    const int father_node = tree_idx * node_num_per_tree 
                  + (1 << (depth - 2)) - 1 + (leaf_idx >> 1);
    if (leaf_idx%2 == 0) {
      mu_data[index] = dp_data[father_node] * d_data[father_node];
    } else {
      mu_data[index] = dp_data[father_node] * (1 - d_data[father_node]);
    }
  }
}

template <typename Dtype>
__global__ void DecisionForestGetProduct(const int nthreads, 
			const Dtype* mu_data, const Dtype* weight_data,
			Dtype* temp, const int num_output, const int N,
            const int tree_num, const int leaf_num_per_tree){
  CUDA_KERNEL_LOOP(index, nthreads){
	const int y = index / (N * tree_num * leaf_num_per_tree);
	const int n = index / (tree_num * leaf_num_per_tree) % N;
	const int t = index / leaf_num_per_tree % tree_num;
	const int l = index % leaf_num_per_tree;
	temp[index] = mu_data[(n * tree_num + t) * leaf_num_per_tree + l]
				 * weight_data[(y * tree_num + t) * leaf_num_per_tree + l];
  }
}

template <typename Dtype>
__global__ void DecisionForestGetProb(const int nthreads, 
			Dtype* temp, Dtype* prob, const int num_output, 
          const int N, const int tree_num, const int leaf_num_per_tree,
          const int gap){
  CUDA_KERNEL_LOOP(index, nthreads){
	const int y = index / (N * tree_num * leaf_num_per_tree);
	const int n = index / (tree_num * leaf_num_per_tree) % N;
	const int t = index / leaf_num_per_tree % tree_num;
	const int l = index % leaf_num_per_tree;
		
	if(l + gap < leaf_num_per_tree){
		temp[index] += temp[index + gap];
	}
	if(gap == 1 && l == 0){
		prob[(n * tree_num + t) * num_output + y] = temp[index];
	}
  }
}

template <typename Dtype>
__global__ void DecisionForestLossGPU(const int nthreads,
          const Dtype* top_data, const Dtype* label, Dtype* loss,
		  const int N, const int num_output) {

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int label_value = static_cast<int>(label[index]);
    loss[index] = -log(top_data[index * num_output + label_value] + eps);
  }
}

template <typename Dtype>
void DecisionForestLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  Dtype* d = this->d_.mutable_gpu_data();
  Dtype* mu = this->mu_.mutable_gpu_data();
  Dtype* dp = this->temp_.mutable_gpu_data();
  Dtype* prob = this->prob_.mutable_gpu_data();
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  Dtype* temp_prob = this->temp_prob_.mutable_gpu_data();
  Dtype* temp_sum = this->temp_sum_.mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();

  DecisionForestGetD<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()),
      CAFFE_CUDA_NUM_THREADS>>>(bottom[0]->count(), d, bottom_data);

  for(int i = 1; i < leaf_num_per_tree_; i <<= 1){
    DecisionForestGetDP<Dtype><<<CAFFE_GET_BLOCKS(N_ * node_num_total_),
      CAFFE_CUDA_NUM_THREADS>>>(N_ * node_num_total_, 
          d, dp, i, N_, num_output_, depth_, tree_num_);
  }

  DecisionForestGetMu<Dtype><<<CAFFE_GET_BLOCKS(N_ * leaf_num_total_),
      CAFFE_CUDA_NUM_THREADS>>>(N_ * leaf_num_total_,
          d, mu, dp, N_, num_output_, depth_, tree_num_);

  DecisionForestGetProduct<Dtype><<<CAFFE_GET_BLOCKS(N_ * num_output_ * leaf_num_total_),
      CAFFE_CUDA_NUM_THREADS>>>(N_ * num_output_ * leaf_num_total_, 
			mu, weight,temp_prob, num_output_, N_,tree_num_, leaf_num_per_tree_);

  int xx = leaf_num_per_tree_, yy = 1;
  if ((xx>>16) > 0) {yy <<= 16; xx >>= 16;}
  if ((xx>>8) > 0) {yy <<= 8; xx >>= 8;}
  if ((xx>>4) > 0) {yy <<= 4; xx >>= 4;}
  if ((xx>>2) > 0) {yy <<= 2; xx >>= 2;}
  if ((xx>>1) > 0) {yy <<= 1; xx >>= 1;}
  const int high_bit = yy;
  for(int i = high_bit; i > 0; i >>= 1){
    DecisionForestGetProb<Dtype><<<CAFFE_GET_BLOCKS(N_ * num_output_ * leaf_num_total_),
      CAFFE_CUDA_NUM_THREADS>>>(N_ * num_output_ * leaf_num_total_, 
			temp_prob, prob, num_output_, N_, tree_num_, leaf_num_per_tree_, i);
  }

  caffe_gpu_gemm (CblasNoTrans, CblasTrans, N_, num_output_, leaf_num_total_,
               (Dtype)1. / tree_num_, mu, weight, (Dtype)0., top[1]->mutable_gpu_data());

  DecisionForestLossGPU<Dtype><<<CAFFE_GET_BLOCKS(N_),
      CAFFE_CUDA_NUM_THREADS>>>(N_, top[1]->gpu_data(), label, loss_data, N_, num_output_);

  Dtype loss;
  caffe_gpu_asum (N_ * tree_num_, loss_data, &loss);
  top[0]->mutable_cpu_data()[0] = loss / (N_ * tree_num_);
}

template <typename Dtype>
__global__ void DecisionForestBackwardGPU(const int nthreads,
          const Dtype* d_data, const Dtype* mu_data, Dtype* Am_data, 
          const Dtype* prob_data, const Dtype* weight_data, const Dtype* label,
          Dtype* bottom_diff, const int N, const int num_output,const int selected_tree,
          const int depth, const int tree_num, const int high_bit) {
   
  const int leaf_num_per_tree = 1 << (depth - 1);
  const int node_num_per_tree = leaf_num_per_tree - 1;
  const int leaf_num_total = leaf_num_per_tree * tree_num;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / node_num_per_tree;
    const int l = index % node_num_per_tree;
	const int label_value = static_cast<int>(label[n]);
    const int idx = (n * tree_num + selected_tree) * node_num_per_tree + l;

    if (l >= (1 << (depth - 2)) - 1) {
      const int childl_mu = (n * tree_num + selected_tree) 
                        * leaf_num_per_tree 
                        + ((l - (1 << (depth - 2)) + 1) << 1);
      const int childr_mu = childl_mu + 1;
      const int childl_weight = label_value * leaf_num_total 
                        + selected_tree * leaf_num_per_tree 
                        + ((l - (1 << (depth - 2)) + 1) << 1);
      const int childr_weight = childl_weight + 1;
      Am_data[idx] = (mu_data[childl_mu] * weight_data[childl_weight] 
                     + mu_data[childr_mu] * weight_data[childr_weight])
                     / (prob_data[(n * tree_num + selected_tree) * num_output + label_value] + eps);
      bottom_diff[idx] = (mu_data[childr_mu] * weight_data[childr_weight] * d_data[idx]
                - mu_data[childl_mu] * weight_data[childl_weight] * (1 - d_data[idx]))
                / (prob_data[(n * tree_num + selected_tree) * num_output + label_value] + eps);
    } else if ((high_bit << 1) > l + 1 && (high_bit & (l + 1))) {
      Am_data[idx] = Am_data[idx + l + 1] + Am_data[idx + l + 2];
      bottom_diff[idx] = Am_data[idx + l + 2] * d_data[idx] 
                          - Am_data[idx + l + 1] * (1 - d_data[idx]); 
    }
  }
}

template <typename Dtype>
__global__ void DecisionForestAddWeightGPU(const int nthreads,
          const Dtype* mu_data, const Dtype* prob_data, const Dtype* label,
          const Dtype* weight_data, Dtype* new_weight_data, const int N, 
          const int num_output, const int depth, const int tree_num) {
   
  const int leaf_num_per_tree = 1 << (depth - 1);
  
  CUDA_KERNEL_LOOP(index, nthreads) {
	const int y = index / (tree_num * leaf_num_per_tree);
    const int t = index / leaf_num_per_tree % tree_num;
    const int l = index % leaf_num_per_tree;

    for(int n = 0; n < N; n++){
      const int label_value = static_cast<int>(label[n]);
      if (label_value == y) {
        new_weight_data[index] += weight_data[index] 
                    * mu_data[(n * tree_num + t) * leaf_num_per_tree + l]
                    / (prob_data[(n * tree_num + t) * num_output + y] + eps);
      }
    }
  }
}

template <typename Dtype>
__global__ void DecisionForestGetWeightSum(const int nthreads,
          Dtype* new_weight_data, const int num_output, 
          const int leaf_num_total, const int gap) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int y = index / leaf_num_total;
    if (y + gap < num_output) {
      new_weight_data[index] += new_weight_data[index + gap * leaf_num_total];
    }
  }
}

template <typename Dtype>
__global__ void DecisionForestUpdateWeight(const int nthreads,
          Dtype* weight_data, Dtype* new_weight_data, 
          const int num_output, const int leaf_num_total) {
  CUDA_KERNEL_LOOP (index, nthreads) {
    const int l = index % leaf_num_total;
    weight_data[index] /= new_weight_data[l];
  }
}


template <typename Dtype>
__global__ void DecisionForestUpdateWeightGPU(const int nthreads,
          Dtype* weight_data, Dtype* new_weight_data,
          const int num_output, const int leaf_num_total) {
  int xx = num_output, yy = 1;
  if ((xx>>16) > 0) {yy <<= 16; xx >>= 16;}
  if ((xx>>8) > 0) {yy <<= 8; xx >>= 8;}
  if ((xx>>4) > 0) {yy <<= 4; xx >>= 4;}
  if ((xx>>2) > 0) {yy <<= 2; xx >>= 2;}
  if ((xx>>1) > 0) {yy <<= 1; xx >>= 1;}
  const int high_bit = yy;
  
  for (int i = high_bit; i > 0; i >>= 1) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      const int y = index / leaf_num_total;
      if (y + i < num_output) {
        new_weight_data[index] += new_weight_data[index + i * leaf_num_total];
      }
    }
    __syncthreads();
  }

  CUDA_KERNEL_LOOP (index, nthreads) {
    const int l = index % leaf_num_total;
    weight_data[index] /= new_weight_data[l];
  }
}

template <typename Dtype>
void DecisionForestLayer<Dtype>::COPY(int n, const Dtype* from, vector<Dtype> &to){
    to.clear();
	Dtype* to_ = new Dtype[n];
	cudaMemcpy(to_, from, n*sizeof(Dtype), cudaMemcpyDeviceToHost);
	for(int i = 0; i < n; i++){
		to.push_back(to_[i]);
	}
	delete[] to_;
}

template <typename Dtype>
void DecisionForestLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  const Dtype* d = this->d_.gpu_data();
  const Dtype* mu = this->mu_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const Dtype* prob = this->prob_.gpu_data();
  Dtype* weight = this->blobs_[0]->mutable_gpu_data();
  Dtype* Am = this->temp_.mutable_gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  if(!change_weight_){
    // calculate bottom_diff
    const int nthreads = N_ * node_num_per_tree_;
    const int selected_tree = caffe_rng_rand() % tree_num_;
    caffe_gpu_set(temp_.count(), (Dtype)0., Am);
    caffe_gpu_set(bottom[0]->count(), (Dtype)0., bottom_diff);

    for(int i = (1 << (depth_ - 2)); i > 0; i >>= 1){
      DecisionForestBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads,
          d, mu, Am, prob, weight, label, bottom_diff, 
          N_, num_output_, selected_tree, depth_, tree_num_, i);
    }
  }else{
    Dtype* new_weight = this->new_weight_.mutable_gpu_data();

    if(iter_backward_ % mini_batch_num_ == 0){
      caffe_gpu_set(leaf_num_total_ * num_output_, (Dtype)0., new_weight);
    }
	iter_backward_++;
    const int nthreads = num_output_ * leaf_num_total_;

    DecisionForestAddWeightGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, mu, prob, label, weight, new_weight, 
                                        N_, num_output_, depth_,tree_num_);

    if(iter_backward_ % mini_batch_num_ == 0){
      const int nthreads = num_output_ * leaf_num_total_;
      caffe_copy(leaf_num_total_ * num_output_, new_weight, weight);

      int xx = num_output_, yy = 1;
      if ((xx>>16) > 0) {yy <<= 16; xx >>= 16;}
      if ((xx>>8) > 0) {yy <<= 8; xx >>= 8;}
      if ((xx>>4) > 0) {yy <<= 4; xx >>= 4;}
      if ((xx>>2) > 0) {yy <<= 2; xx >>= 2;}
      if ((xx>>1) > 0) {yy <<= 1; xx >>= 1;}
      const int high_bit = yy;
      for(int i = high_bit; i > 0; i >>= 1){
        DecisionForestGetWeightSum<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
          CAFFE_CUDA_NUM_THREADS>>>(nthreads,
                     new_weight, num_output_, leaf_num_total_, i);
      }

      DecisionForestUpdateWeight<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads,
                   weight, new_weight, num_output_, leaf_num_total_);
    }

  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DecisionForestLayer);

}  // namespace caffe
