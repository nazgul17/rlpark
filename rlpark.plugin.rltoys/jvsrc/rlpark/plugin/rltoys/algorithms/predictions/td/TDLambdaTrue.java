package rlpark.plugin.rltoys.algorithms.predictions.td;

import rlpark.plugin.rltoys.algorithms.traces.EligibilityTraceAlgorithm;
import rlpark.plugin.rltoys.algorithms.traces.Traces;
import rlpark.plugin.rltoys.math.vector.RealVector;

/**
 * Created by Marco Tamassia.
 */
public class TDLambdaTrue extends TDLambda implements EligibilityTraceAlgorithm {
    private static final long serialVersionUID = 4861872863595788819L;
    private double gamma_t;
    protected double v_old;

    public TDLambdaTrue(double alpha, double gamma, double lambda, int nbFeatures, Traces prototype) {
        super(lambda, gamma, alpha, nbFeatures, prototype);
    }

    public double initEpisode() {
        v_old = 0;
        return super.initEpisode();
    }

    public double update(RealVector x_t, RealVector x_tp1, double r_tp1, double gamma_tp1) {
        v_t = v.dotProduct(x_t);
        double v_tp1 = v.dotProduct(x_tp1);
        delta_t = r_tp1 + gamma_tp1 * v_tp1 - v_t;

        e.update(gamma_t * lambda, x_t, (1.0 - alpha_v * gamma_t * lambda * e.vect().dotProduct(x_t)));
        v.addToSelf(-alpha_v * (v_t - v_old), x_t).addToSelf(alpha_v * (delta_t + v_t - v_old), e.vect());

        v_old = v_tp1;
        gamma_t = gamma_tp1;
        return delta_t;
    }

    @Override
    public Traces traces() {
        return e;
    }
}
