// Meep 1.35.0 version-pinned adapter. No installed library is modified.
// Scale existing Lorentz forcing arrays while retaining every P/P_prev state.
#include <meep.hpp>
#include <vector>
#include <set>
#include <string>
#include <cmath>
#include <stdexcept>
#include <memory>

namespace {
thread_local std::string error;
struct Array {meep::realnum *address; std::vector<meep::realnum> baseline;};
struct State {meep::polarization_state *address; const meep::susceptibility *sus; void *data;};
struct Handle {meep::fields *fields; std::vector<Array> arrays; std::vector<State> states;};
void check(Handle *h) {
  if(!h || !h->fields || h->states.empty()) throw std::runtime_error("Invalid Lorentz adapter handle");
  for(auto &s:h->states) if(s.address->s!=s.sus || s.address->data!=s.data)
    throw std::runtime_error("Lorentz state changed after adapter creation; recreate adapter after a native reset");
}
}
extern "C" {
const char *lorentz_sigma_error(){return error.c_str();}
int lorentz_realnum_size(){return sizeof(meep::realnum);}
void *lorentz_sigma_create(void *pointer) {
  try {
    auto *f=static_cast<meep::fields*>(pointer);if(!f)throw std::runtime_error("Null Meep fields");
    auto h=std::make_unique<Handle>();h->fields=f;
    std::set<meep::realnum*> seen_arrays;std::set<int> ids;
    for(int i=0;i<f->num_chunks;i++) {
      auto *chunk=f->chunks[i];
      for(auto *p=chunk->pol[meep::E_stuff];p;p=p->next) {
        if(!dynamic_cast<const meep::lorentzian_susceptibility*>(p->s))
          throw std::runtime_error("Only native Lorentz susceptibilities are supported");
        ids.insert(p->s->get_id());h->states.push_back({p,p->s,p->data});
        for(int c=0;c<meep::NUM_FIELD_COMPONENTS;c++)for(int d=0;d<5;d++){
          auto *a=p->s->sigma[c][d];if(!a || !seen_arrays.insert(a).second)continue;
          Array ar;ar.address=a;ar.baseline.assign(a,a+p->s->ntot);
          for(auto v:ar.baseline)if(!std::isfinite(v))throw std::runtime_error("Nonfinite baseline sigma");
          h->arrays.push_back(std::move(ar));
        }
      }
    }
    if(ids.size()!=1 || h->arrays.empty() || h->states.empty()){
      throw std::runtime_error("Require exactly one initialized electric Lorentz susceptibility ID");
    }
    return h.release();
  }catch(const std::exception &e){error=e.what();return nullptr;}
}
int lorentz_sigma_scale(void *pointer,double factor){
  try{
    auto *h=static_cast<Handle*>(pointer);check(h);
    if(!std::isfinite(factor)||factor<=0)throw std::runtime_error("Sigma factor must be finite positive");
    for(auto &ar:h->arrays)for(size_t i=0;i<ar.baseline.size();i++)ar.address[i]=ar.baseline[i]*factor;
    return 0;
  }catch(const std::exception &e){error=e.what();return -1;}
}
int lorentz_sigma_array_count(void *pointer){
  try{auto *h=static_cast<Handle*>(pointer);check(h);return h->arrays.size();}
  catch(const std::exception &e){error=e.what();return -1;}
}
int lorentz_sigma_state_count(void *pointer){
  try{auto *h=static_cast<Handle*>(pointer);check(h);return h->states.size();}
  catch(const std::exception &e){error=e.what();return -1;}
}
void lorentz_sigma_destroy(void *pointer){delete static_cast<Handle*>(pointer);}
}
