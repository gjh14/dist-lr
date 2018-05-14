#include <iostream>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <thread>
#include "ps/ps.h"

#include "lr.h"
#include "util.h"
#include "data_iter.h"

const int kSyncMode = -1;

template <typename Val>
class KVStoreDistServer {
public:
  KVStoreDistServer() {
    using namespace std::placeholders;
    ps_server_ = new ps::KVServer<float>(0);
    ps_server_->set_request_handle(
      std::bind(&KVStoreDistServer::DataHandle, this, _1, _2, _3));

    sync_mode_ = !strcmp(ps::Environment::Get()->find("SYNC_MODE"), "1");
    learning_rate_ = distlr::ToFloat(ps::Environment::Get()->find("LEARNING_RATE"));

    std::string mode = sync_mode_ ? "sync" : "async";
    std::cout << "Server mode: " << mode << std::endl;
  }

  ~KVStoreDistServer() {
    if (ps_server_) {
      delete ps_server_;
    }
  }

private:

  // threadsafe
  void DataHandle(const ps::KVMeta& req_meta,
                  const ps::KVPairs<Val>& req_data,
                  ps::KVServer<Val>* server) {
    size_t n = req_data.keys.size();
    if (req_meta.push) {
      CHECK_EQ(n, req_data.vals.size());
      if (weights_.empty()) {
        std::cout << "Init weight " << n << std::endl;
        weights_.resize(n);
        for (size_t i = 0; i < n; ++i) {
          weights_[i] = req_data.vals[i];
        }
        server->Response(req_meta);
      } else if (sync_mode_) {
        if (merge_buf_.vals.empty()) {
          merge_buf_.vals.resize(n, 0);
        }

        for (size_t i = 0; i < n; ++i) {
          merge_buf_.vals[req_data.keys[i]] += req_data.vals[i];
        }

        merge_buf_.request.push_back(req_meta);
        if (merge_buf_.request.size() == (size_t)ps::NumWorkers()) {
          // update the weight
          for (size_t i = 0; i < n; ++i) {
            weights_[i] -= learning_rate_ * merge_buf_.vals[i] / merge_buf_.request.size();
          }
          for (const auto& req : merge_buf_.request) {
            server->Response(req);
          }
          merge_buf_.request.clear();
          merge_buf_.vals.clear();
        }
      } else { // async push
        for (size_t i = 0; i < n; ++i) {
          weights_[req_data.keys[i]] -= learning_rate_ * req_data.vals[i];
        }
        server->Response(req_meta);
      }
    } else { // pull
      ps::KVPairs<Val> response;
      response.keys = req_data.keys;
      response.vals.resize(n);
      for (size_t i = 0; i < n; ++i) {
        response.vals[i] = weights_[req_data.keys[i]];
      }
      server->Response(req_meta, response);
    }
  }

  bool sync_mode_;
  float learning_rate_;

  struct MergeBuf {
    std::vector<ps::KVMeta> request;
    std::vector<Val> vals;
  };

  std::vector<Val> weights_;
  MergeBuf merge_buf_;
  ps::KVServer<float>* ps_server_;
};

void StartServer() {
  if (!ps::IsServer()) {
    return;
  }
  auto server = new KVStoreDistServer<float>();
  ps::RegisterExitCallback([server](){ delete server; });
}

void RunWorker() {
  if (!ps::IsWorker()) {
    return;
  }

  std::string root = ps::Environment::Get()->find("DATA_DIR");
  int num_feature_dim =
    distlr::ToInt(ps::Environment::Get()->find("NUM_FEATURE_DIM"));

  int rank = ps::MyRank();
  ps::KVWorker<float>* kv = new ps::KVWorker<float>(0);
  distlr::LR lr = distlr::LR(num_feature_dim);
  lr.SetKVWorker(kv);

  if (rank == 0) {
    auto vals = lr.GetWeight();
    std::vector<ps::Key> keys(vals.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      keys[i] = i;
    }
    kv->Wait(kv->Push(keys, vals));
  }
  ps::Postoffice::Get()->Barrier(ps::kWorkerGroup);

  std::cout << "Worker[" << rank << "]: start working..." << std::endl;
  int num_iteration = distlr::ToInt(ps::Environment::Get()->find("NUM_ITERATION"));
  int batch_size = distlr::ToInt(ps::Environment::Get()->find("BATCH_SIZE"));
  // int test_interval = distlr::ToInt(ps::Environment::Get()->find("TEST_INTERVAL"));

  for (int i = 0; i < num_iteration; ++i) {
    static std::string filename = root + "/train/part-00" + std::to_string(rank + 1);
    static distlr::DataIter iter(filename, num_feature_dim);
    iter.Init();
    lr.Train(iter, i + 1, batch_size);

    /* if (rank == 0 and (i + 1) % test_interval == 0) {
      static std::string filename = root + "/test/part-001";
      static distlr::DataIter test_iter(filename, num_feature_dim);
      test_iter.Init();
      lr.Test(test_iter, i + 1);
    } */
  }
  std::string modelfile = root + "/models/part-00" + std::to_string(rank + 1);
  lr.SaveModel(modelfile);
}

int main() {
  StartServer();

  ps::Start();
  RunWorker();

  ps::Finalize();
  return 0;
}

