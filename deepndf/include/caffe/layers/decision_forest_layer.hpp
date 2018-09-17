#ifndef CAFFE_DECISION_FOREST_LAYER_HPP_
#define CAFFE_DECISION_FOREST_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes a decision forest
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class DecisionForestLayer : public LossLayer<Dtype> {
 public:
  explicit DecisionForestLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DecisionForest"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }
  

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  void COPY(int n, const Dtype* from, vector<Dtype> &to);

  int N_;
  int tree_num_;
  int depth_;
  int node_num_per_tree_;
  int leaf_num_per_tree_;
  int node_num_total_;
  int leaf_num_total_;
  int num_output_;
  int df_axis_;
  int iter_backward_;
  int mini_batch_num_;
  bool change_weight_;
  /// probability distribution in leaves nodes.
  // Blob<Dtype> prediction_node_;
  Blob<Dtype> d_;
  /// routing function.
  Blob<Dtype> mu_;
  /// used to calculate routing function and Am.
  Blob<Dtype> temp_;
  /// used to calculate prob.
  Blob<Dtype> temp_prob_;
  Blob<Dtype> temp_sum_;
  /// prediction result.
  Blob<Dtype> prob_;

  Blob<Dtype> new_weight_;

};

}  // namespace caffe

#endif  // CAFFE_INNER_PRODUCT_LAYER_HPP_
