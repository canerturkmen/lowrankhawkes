import itertools
import pickle
import time
import tarfile
import sys
import uuid
import warnings
from collections import OrderedDict
from pathlib import Path

import hawkeslib as hl
import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import normalized_mutual_info_score as nmi
from tick.hawkes import HawkesExpKern, HawkesEM
from tick.hawkes.inference.hawkes_cumulant_matching import HawkesCumulantMatching

sys.path.append("../")

from lrhp.neumann import TruncatedNeumannEstimator
from lrhp.util import hawkeslib_data_to_tick


class Experiment:
    def __init__(self, name, fit_args, model, n_clusters=10):
        self.name = name
        self.fit_args = fit_args
        self.model = model
        self.n_clusters = n_clusters

    def get_spectral_clustering(self, P, is_W=False):
        if is_W:
            P = P.T.dot(P)

        P = np.clip(P, a_min=0, a_max=None)

        return SpectralClustering(
            n_clusters=self.n_clusters, affinity="precomputed"
        ).fit_predict(P)

    def get_kmeans_clustering(self, P, is_W=False):
        if not is_W:
            P = np.clip(P, a_min=0, a_max=None)
            P = NMF(n_components=self.n_clusters).fit_transform(P)
        else:
            P = P.T

        return KMeans(n_clusters=self.n_clusters).fit_predict(P)

    @staticmethod
    def _get_tick_phi(model):
        phi_getter = {
            HawkesEM: lambda m: m.kernel.sum(-1),
            HawkesCumulantMatching: lambda m: m.solution,
            HawkesExpKern: lambda m: m.adjacency,
        }
        for k, v in phi_getter.items():
            if isinstance(model, k):
                return v(model)
        raise ValueError("Given model not recognized.")

    def run(self):
        start_time = time.time()

        if "tick" in str(type(self.model)):
            self.model.fit(*self.fit_args)
            phi = self._get_tick_phi(self.model)
            fitting_time = time.time() - start_time

            sc = self.get_spectral_clustering(phi)
            km = self.get_kmeans_clustering(phi)
        else:
            assert isinstance(self.model, TruncatedNeumannEstimator)
            W, _ = self.model.fit(**self.fit_args)
            phi = np.clip(W.T.dot(W), a_min=0, a_max=None).astype(np.float64)
            fitting_time = time.time() - start_time

            sc = self.get_spectral_clustering(W, is_W=True)
            km = self.get_kmeans_clustering(W, is_W=True)

        return fitting_time, phi, sc, km


def main(n_clusters=10):
    with tarfile.open("../data/synthetic_hawkes_data.tar.gz", "r:gz") as tar:
        fp = tar.extractfile("synthetic_hawkes_data")
        mu, Phi, beta, t, c = pickle.load(fp)
        fp.close()

    test_ix = len(t) // 2
    t1, c1 = (x[:test_ix] for x in (t, c))
    t2, c2 = (x[test_ix:] for x in (t, c))
    t2 -= t1[-1]

    tickd = hawkeslib_data_to_tick(t1, c1)
    SHORT_DATA_LENGTH = 2_000_000
    tickd_short = hawkeslib_data_to_tick(t1[:SHORT_DATA_LENGTH], c1[:SHORT_DATA_LENGTH])

    baseline_experiments = [
        Experiment(
            name="NPHC",
            fit_args=[tickd],
            model=HawkesCumulantMatching(
                integration_support=1.0,
                verbose=True,
                C=1.,
                max_iter=1000,
            ),
            n_clusters=n_clusters,
        ),
        Experiment(
            name="Hawkes-LS",
            fit_args=[tickd_short],
            model=HawkesExpKern(
                decays=1.,
                gofit="least-squares",
                C=1,
                solver="gd"
            ),
            n_clusters=n_clusters,
        ),
        Experiment(
            name="Hawkes-EM",
            fit_args=[tickd_short],
            model=HawkesEM(
                kernel_support=10.,
                kernel_size=2,
                verbose=True,
                print_every=10,
            ),
            n_clusters=n_clusters,
        )
    ]

    neumann_experiments = [
        Experiment(
            name="LRHP-GD",
            fit_args=dict(
                t=t1, c=c1, num_epochs=int(1e3), learning_rate=1e-2
            ),
            model=TruncatedNeumannEstimator(rank=n_clusters, is_nmf=False),
            n_clusters=n_clusters,
        ),
        Experiment(
            name="LRHP-GD (NMF)",
            fit_args=dict(
                t=t1, c=c1, num_epochs=int(5e4), learning_rate=2e-1
            ),
            model=TruncatedNeumannEstimator(rank=10, is_nmf=True),
            n_clusters=n_clusters,
        ),
    ]

    # get original clusters via NMF
    nmf = NMF(n_components=n_clusters)
    Wo = nmf.fit_transform(Phi)
    orig_clus = KMeans(n_clusters=n_clusters).fit_predict(Wo)

    # run experiments
    warnings.simplefilter("ignore")

    all_results = []

    for exp in baseline_experiments + neumann_experiments:
        time_taken, phi, sc, km = exp.run()
        results = OrderedDict(
            name=exp.name,
            n_clusters=n_clusters,
            time_taken=time_taken,
            pred_ll=hl.MultivariateExpHawkesProcess.log_likelihood_with_params(
                t2, c2, mu, np.clip(phi, a_min=0, a_max=None), beta
            ) / len(t2),
            sc_nmi=nmi(sc, orig_clus),
            km_nmi=nmi(km, orig_clus),
        )

        reslist = list(results.values())
        all_results.append(reslist)
        print(reslist)

    # write out results
    out_path = Path("./outputs/")
    out_path.mkdir(exist_ok=True)

    with open(out_path / f"{str(uuid.uuid4())[:7]}", "w") as fp:
        for r in all_results:
            print(",".join(map(str, r)), file=fp)


if __name__ == "__main__":
    # Parallel(n_jobs=36)(delayed(main)() for _ in range(20))
    main()
