package rlpark.plugin.rltoys.algorithms.predictions.td;

import rlpark.plugin.rltoys.algorithms.traces.EligibilityTraceAlgorithm;
import rlpark.plugin.rltoys.algorithms.traces.Traces;
import rlpark.plugin.rltoys.math.vector.MutableVector;
import rlpark.plugin.rltoys.math.vector.RealVector;
import rlpark.plugin.rltoys.math.vector.pool.VectorPool;
import rlpark.plugin.rltoys.math.vector.pool.VectorPools;

/**
 * Created by Marco Tamassia.
 */
public class EmphaticTDLambdaBeta extends TD implements EligibilityTraceAlgorithm {
    /**
     * A variant of emphatic TD where parameter beta reduces the magnitude of the follow-on trace, which mitigates the
     * high variance potentially affecting the original algorithm.
     *
     * Ref: Hallak, Assaf, et al. "Generalized emphatic temporal difference learning: Bias-variance analysis." arXiv preprint arXiv:1509.05172 (2015).
     * Link: http://arxiv.org/pdf/1509.05172
     */

    protected Traces e;      // eligibility trace vector
    protected double F;        // scalar memory for the emphasis algorithm
    protected double D, gamma; // auxiliary saved scalars from one step to the next
    protected int n;           //dimensionality of the vectors

    public EmphaticTDLambdaBeta(double alpha, double gamma, int nbFeatures, Traces prototype) {
        super(gamma, alpha, nbFeatures);
        e = prototype.newTraces(nbFeatures);
        F = 0;
        D = 0;
        gamma = 0;
        this.n = n;
    }

    /**
     * Adds knowledge to internal representation.
     * @param alpha_t learning step, in [0,1]
     * @param I_t set of interest for S_t, in [0,1]
     * @param lambda_t eligibility trace parameter, in [0,1]
     * @param x_t feature vector corresponding to action A_t in state S_t
     * @param rho_t ratio of target policy to behaviour policy
     * @param R_tp1 transient reward
     * @param x_tp1 expected next state feature vector corresponding to a \in A and S_t+1
     * @param gamma_tp1 discount factor, in [0,1]
     * @param beta_tp1 decay rate of the importance-sampling ratio, in [0,1]
     */
    public void learn(double alpha_t, double I_t, double lambda_t, RealVector x_t, double rho_t, double R_tp1, RealVector x_tp1, double gamma_tp1, double beta_tp1) {
        VectorPool pool = VectorPools.pool(e.vect());
        double delta = R_tp1 + gamma_tp1 * v.dotProduct(x_tp1) - v.dotProduct(x_t);
        F = beta_tp1 * F + I_t;
        double M = lambda_t * I_t + (1 - lambda_t) * F;
        double S = rho_t * alpha_t * M * (1 - rho_t * gamma * lambda_t * e.vect().dotProduct(x_t));

        e.update(rho_t * gamma * lambda_t, x_t, S);
        MutableVector deltaE = pool.newVector(e.vect()).mapMultiplyToSelf(delta);
        deltaE.addToSelf(D, pool.newVector(e.vect()).subtractToSelf(x_t.mapMultiply(rho_t * alpha_t * M)));
        v.addToSelf(deltaE);
        D = deltaE.dotProduct(x_tp1);

        F *= rho_t;
        gamma = gamma_tp1;
    }

    @Override
    public Traces traces() {
        return e;
    }

}
