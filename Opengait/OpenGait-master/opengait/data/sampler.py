import math
import random
import torch
import torch.distributed as dist
import torch.utils.data as tordata

def _dist_on():
    """분산 모드가 초기화돼 있으면 True 반환"""
    return dist.is_available() and dist.is_initialized()


class TripletSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size, batch_shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        if len(self.batch_size) != 2:
            raise ValueError("batch_size should be (P x K) not {}".format(batch_size))
        self.batch_shuffle = batch_shuffle

        # ----- 분산 여부에 따라 rank / world_size 설정 -----
        self.world_size = dist.get_world_size() if _dist_on() else 1
        self.rank       = dist.get_rank()       if _dist_on() else 0

        if (self.batch_size[0]*self.batch_size[1]) % self.world_size != 0:
            raise ValueError("World size ({}) is not divisible by batch_size ({} x {})".format(
                self.world_size, batch_size[0], batch_size[1]))
        self.rank = dist.get_rank()

    def __iter__(self):
        while True:
            sample_indices = []
            pid_list = sync_random_sample_list(
                self.dataset.label_set, self.batch_size[0])

            for pid in pid_list:
                indices = self.dataset.indices_dict[pid]
                indices = sync_random_sample_list(
                    indices, k=self.batch_size[1])
                sample_indices += indices

            if self.batch_shuffle:
                sample_indices = sync_random_sample_list(
                    sample_indices, len(sample_indices))

            total_batch_size = self.batch_size[0] * self.batch_size[1]
            total_size = int(math.ceil(total_batch_size /
                                       self.world_size)) * self.world_size
            sample_indices += sample_indices[:(
                total_batch_size - len(sample_indices))]

            sample_indices = sample_indices[self.rank:total_size:self.world_size]
            yield sample_indices

    def __len__(self):
        return len(self.dataset)


def sync_random_sample_list(obj_list, k, common_choice=False):
    # ── 난수 선택 ───────────────────────────────────────────
    if common_choice or len(obj_list) < k:
        idx = torch.tensor(random.choices(range(len(obj_list)), k=k))
    else:
        idx = torch.randperm(len(obj_list))[:k]

    # (선택) GPU ↑
    if torch.cuda.is_available():
        idx = idx.cuda()

    # ── 분산 브로드캐스트 우회 ──────────────────────────────
    if _dist_on():
        dist.broadcast(idx, src=0)

    idx = idx.cpu().tolist()  # CPU 로 돌려놓고 list 화
    return [obj_list[i] for i in idx]


class InferenceSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        self.size = len(dataset)
        indices = list(range(self.size))

        world_size = dist.get_world_size() if _dist_on() else 1
        rank       = dist.get_rank()       if _dist_on() else 0

        if _dist_on() and batch_size % world_size != 0:
            raise ValueError("World size ({}) is not divisible by batch_size ({})".format(
                world_size, batch_size))

        if batch_size != 1:
            complement_size = math.ceil(self.size / batch_size) * \
                batch_size
            indices += indices[:(complement_size - self.size)]
            self.size = complement_size

        batch_size_per_rank = int(self.batch_size / world_size)
        indx_batch_per_rank = []

        for i in range(int(self.size / batch_size_per_rank)):
            indx_batch_per_rank.append(
                indices[i*batch_size_per_rank:(i+1)*batch_size_per_rank])

        self.idx_batch_this_rank = indx_batch_per_rank[rank::world_size]

    def __iter__(self):
        yield from self.idx_batch_this_rank

    def __len__(self):
        return len(self.dataset)


class CommonSampler(tordata.sampler.Sampler):
    def __init__(self,dataset,batch_size,batch_shuffle):

        self.dataset = dataset
        self.size = len(dataset)
        self.batch_size = batch_size
        if isinstance(self.batch_size,int)==False:
            raise ValueError(
                "batch_size shoude be (B) not {}".format(batch_size))
        self.batch_shuffle = batch_shuffle
        
        self.world_size = dist.get_world_size() if _dist_on() else 1
        self.rank       = dist.get_rank()       if _dist_on() else 0

        if _dist_on() and self.batch_size % self.world_size != 0:
            raise ValueError(
                f"World size ({self.world_size}) is not divisible by batch_size ({batch_size})")
    
    def __iter__(self):
        while True:
            indices_list = list(range(self.size))
            sample_indices = sync_random_sample_list(
                    indices_list, self.batch_size, common_choice=True)
            total_batch_size =  self.batch_size
            total_size = int(math.ceil(total_batch_size /
                                       self.world_size)) * self.world_size
            sample_indices += sample_indices[:(
                total_batch_size - len(sample_indices))]
            sample_indices = sample_indices[self.rank:total_size:self.world_size]
            yield sample_indices

    def __len__(self):
        return len(self.dataset)

# **************** For GaitSSB ****************
# Fan, et al: Learning Gait Representation from Massive Unlabelled Walking Videos: A Benchmark, T-PAMI2023
import random
class BilateralSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size, batch_shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_shuffle = batch_shuffle

        self.world_size = dist.get_world_size() if _dist_on() else 1
        self.rank       = dist.get_rank()       if _dist_on() else 0

        self.dataset_length = len(self.dataset)
        self.total_indices = list(range(self.dataset_length))

    def __iter__(self):
        random.shuffle(self.total_indices)
        count = 0
        batch_size = self.batch_size[0] * self.batch_size[1]
        while True:
            if (count + 1) * batch_size >= self.dataset_length:
                count = 0
                random.shuffle(self.total_indices)

            sampled_indices = self.total_indices[count*batch_size:(count+1)*batch_size]
            sampled_indices = sync_random_sample_list(sampled_indices, len(sampled_indices))

            total_size = int(math.ceil(batch_size / self.world_size)) * self.world_size
            sampled_indices += sampled_indices[:(batch_size - len(sampled_indices))]

            sampled_indices = sampled_indices[self.rank:total_size:self.world_size]
            count += 1

            yield sampled_indices * 2

    def __len__(self):
        return len(self.dataset)