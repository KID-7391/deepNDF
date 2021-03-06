#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/decision_forest_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DecisionForestLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  df_axis_ = bottom[0]->CanonicalAxisIndex(
            this->layer_param_.decision_forest_param().df_axis());
  const int num_input = bottom[0]->count(df_axis_);
  N_ = bottom[0]->count(0, df_axis_);
  tree_num_ = this->layer_param_.decision_forest_param().tree_num();
  depth_ = this->layer_param_.decision_forest_param().depth();
  leaf_num_per_tree_ = 1 << (depth_ - 1);
  node_num_per_tree_ = leaf_num_per_tree_ - 1;
  leaf_num_total_ = tree_num_ * leaf_num_per_tree_;
  node_num_total_ = tree_num_ * node_num_per_tree_;
  num_output_ = this->layer_param_.decision_forest_param().num_output();
  iter_backward_ = 0;
  mini_batch_num_ = this->layer_param_.decision_forest_param().mini_batch_num();
  change_weight_ = this->layer_param_.decision_forest_param().change_weight();
	
  CHECK_EQ(node_num_per_tree_*tree_num_, num_input)
      << "Input size incompatible with decision forest parameters.";
	
  if(this->blobs_.size() > 0){
    LOG(INFO) << "Skipping parameter initialization";
  }else{
    this->blobs_.resize(1);
    // Intialize the leaf nodes
    vector<int> weight_shape(3);
    weight_shape[0] = num_output_;
    weight_shape[1] = tree_num_;
    weight_shape[2] = leaf_num_per_tree_;
    
    this->blobs_[0].reset(new Blob<Dtype> (weight_shape));
    
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.decision_forest_param().weight_filler()));    
    weight_filler->Fill(this->blobs_[0].get());  

  }
	
  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), false);
  this->set_loss (0, 1);

}

template <typename Dtype>
void DecisionForestLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  d_.ReshapeLike(*bottom[0]);
  mu_.Reshape(N_, tree_num_, leaf_num_per_tree_, 1);
  temp_.ReshapeLike(*bottom[0]);
  temp_prob_.Reshape(num_output_, N_, tree_num_, leaf_num_per_tree_);
  temp_sum_.Reshape(N_, tree_num_, num_output_, 1);
  prob_.Reshape(N_, tree_num_, num_output_, 1);
  new_weight_.Reshape(num_output_, tree_num_, leaf_num_per_tree_, 1);
  

  vector<int> top0_shape(2);
  top0_shape[0] = top0_shape[1] = 1;
  top[0]->Reshape(top0_shape);
  if(top.size() >= 2){
	vector<int> top1_shape(2);
    top1_shape[0] = N_;
    top1_shape[1] = num_output_;
    top[1]->Reshape(top1_shape);
  }

  CHECK_EQ(N_ * prob_.count(df_axis_ + 2), bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if decision forest axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
}

template <typename Dtype>
inline float sigmoid(Dtype x){
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
void DecisionForestLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  Dtype* d = this->d_.mutable_cpu_data();
  Dtype* mu = this->mu_.mutable_cpu_data();
  Dtype* dp = this->temp_.mutable_cpu_data();
  Dtype* prob = this->prob_.mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  const int count = bottom[0]->count();
  // get d
  for(int i = 0; i < count; i++){
    d[i] = sigmoid(bottom_data[i]);
  }

  for(int i = 0; i < N_; i++){
    int mu_cnt = 0;
    for(int j = 0; j < tree_num_; j++){
      const int base = i*node_num_total_ + j*node_num_per_tree_;
      dp[base] = 1;
      for(int k = 1; k < node_num_per_tree_; k++){
        if(k%2){
          dp[base + k] = d[base + (k>>1)] * dp[base + (k>>1)];
        }else{
          dp[base + k] = (1 - d[base + (k>>1) - 1]) * dp[base + (k>>1) - 1];
        }
        // calculate routing function
        if(k >= (1<<(depth_ - 2)) - 1){
          mu[i*leaf_num_total_ + mu_cnt++] = 
	 		dp[base + k] * d[base + k];
          mu[i*leaf_num_total_ + mu_cnt++] = 
	 		dp[base + k] * (1- d[base + k]);
        }
      }
    }
  }
	
  // get prediction
  for(int i = 0; i < N_; i++){
    for(int y = 0; y < num_output_; y++){
      prob[i*num_output_ + y] = caffe_cpu_strided_dot<Dtype>(leaf_num_total_, 
                                weight + y * leaf_num_total_, 1, 
                                mu + i*leaf_num_total_, 1);
    }
  }
  caffe_cpu_scale (N_*num_output_, (Dtype)1./tree_num_, prob, prob);

	
  // calculate loss
  Dtype loss = 0;
  for(int i = 0; i < N_; i++){
    const int label_value = static_cast<int>(label[i]);
    loss -= log(prob[i * num_output_ + label_value]);
  }
  top[0]->mutable_cpu_data()[0] = loss / N_;

  if(top.size() == 2){
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void DecisionForestLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* d = this->d_.cpu_data();
  const Dtype* mu = this->mu_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  const Dtype* prob = this->prob_.cpu_data();
  Dtype* weight = this->blobs_[0]->mutable_cpu_data();
  Dtype* new_weight = this->blobs_[1]->mutable_cpu_data();
  Dtype* Am = this->temp_.mutable_cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  /*if(update_iterative_ == -1){
	iter_backward_++;
    // calculate bottom_diff
    for(int i = 0; i < N_; i++){
      int cnt = leaf_num_total_ - 1;
      const int label_value = static_cast<int>(label[i]);
      const int update_tree = caffe_rng_rand() % tree_num_;
      for(int j = tree_num_ - 1; j >= 0 ; j--){
        const int base = i*node_num_total_ + j*node_num_per_tree_;
        if(j == update_tree){
          for(int k = (1<<(depth_-1))-2; k >= 0; k--){
            if(k >= (1<<(depth_-2))-1){
              const int idx_w = label_value * leaf_num_total_ + cnt;
              const int idx_mu = i * leaf_num_total_ + cnt;
              Am[base + k] = (mu[idx_mu] * weight[idx_w] + mu[idx_mu-1] * weight[idx_w - 1]) 
                              / prob[i * num_output_ + label_value];
              bottom_diff[base + k] = mu[idx_mu] * weight[idx_w] * d[base + k] - 
                                      mu[idx_mu - 1] * weight[idx_w - 1] * (1 - d[base + k]);
              cnt -= 2;
            }else{
              Am[base + k] = Am[base + 2*k + 1] + Am[base + 2*k + 2];
              bottom_diff[base + k] = Am[base + 2*k + 2] * d[base + k] - 
                                      Am[base + 2*k + 1] * (1 - d[base + k]);
            }
          }
        }else{
          for(int k = (1<<(depth_-1))-2; k >= 0; k--){
            bottom_diff[base + k] = 0;
          }
		  cnt -= leaf_num_per_tree_;
        }
      }
    }
	  
  }else{
	caffe_set(bottom[0]->count(), (Dtype)0., bottom_diff);
	if((iter_backward_ / mini_batch_num_ + 1) % (update_iterative_ + 1) == 1 &&
	   iter_backward_ % mini_batch_num_ == 0){
	  caffe_set(leaf_num_total_ * num_output_, (Dtype)1./num_output_, weight);
      caffe_set(leaf_num_total_ * num_output_, (Dtype)0., new_weight);
	}
	iter_backward_++;
    if(iter_backward_ % mini_batch_num_){
      for(int i = 0; i < N_; i++){
		const int label_value = static_cast<int>(label[i]);
		for(int l = 0; l < leaf_num_total_; l++){
		  new_weight[label_value * leaf_num_total_ + l] += 
				weight[label_value * leaf_num_total_ + l] * mu[i * leaf_num_total_ + l]
				/ prob[i * num_output_ + label_value];
		}
      }
	}else{
	  // vector<double> vec_debug;
      for(int l = 0; l < leaf_num_total_; l++){
        double sum = 0;
		for(int i = 0; i < num_output_; i++){
          sum += new_weight[i * leaf_num_total_ + l];
		}
		for(int i = 0; i < num_output_; i++){
          new_weight[i * leaf_num_total_ + l] /= sum;
		  // vec_debug.push_back(new_weight[i * leaf_num_total_ + l]);
		}
	  }
      caffe_copy(leaf_num_total_ * num_output_, new_weight, weight);
      caffe_set(leaf_num_total_ * num_output_, (Dtype)0., new_weight);
	}
  }*/
}

#ifdef CPU_ONLY
STUB_GPU(DecisionForestLayer);
#endif

INSTANTIATE_CLASS(DecisionForestLayer);
REGISTER_LAYER_CLASS(DecisionForest);

}  // namespace caffe
