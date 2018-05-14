#ifndef DISTLR_LR_H_
#define DISTLR_LR_H_

#include "data_iter.h"
#include <cstdio>
#include <sys/time.h>

namespace distlr {

class LR {
public:
  explicit LR(int num_feature_dim, float learning_rate=0.001, float C_=1,
              int random_state=0);
  virtual ~LR() {
    if (file)
      fclose(file);
    if (kv_) {
      delete kv_;
    }
  }

  void SetKVWorker(ps::KVWorker<float>* kv);

  void Train(DataIter& iter, int num_iter, int batch_size);

  void Test(DataIter& iter, int num_iter);

  std::vector<float> GetWeight();

  ps::KVWorker<float>* GetKVWorker();

  bool SaveModel(std::string& filename);

  std::string DebugInfo();

private:
  void InitWeight_();

  int Predict_(std::vector<float> feature);

  float Sigmoid_(std::vector<float> feature);

  void PullWeight_();

  void PushGradient_(const std::vector<float>& grad, int num_iter);
  
  void AddTime(timeval& sum, timeval& st, timeval& ed);

  int num_feature_dim_;
  float learning_rate_;
  float C_;

  int random_state_;

  std::vector<ps::Key> keys_;
  std::vector<float> weight_;

  ps::KVWorker<float>* kv_;
  
  float udf_;
  timeval net_, calc_, start_;
  FILE *file;
};

}  // namespace distlr

#endif  // LR_H_
