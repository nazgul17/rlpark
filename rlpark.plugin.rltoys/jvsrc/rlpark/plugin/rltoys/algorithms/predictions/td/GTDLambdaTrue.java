package rlpark.plugin.rltoys.algorithms.predictions.td;

import rlpark.plugin.rltoys.algorithms.traces.EligibilityTraceAlgorithm;
import rlpark.plugin.rltoys.algorithms.traces.Traces;
import rlpark.plugin.rltoys.math.vector.RealVector;
import rlpark.plugin.rltoys.math.vector.implementations.PVector;
import zephyr.plugin.core.api.internal.monitoring.wrappers.Abs;
import zephyr.plugin.core.api.internal.monitoring.wrappers.Squared;
import zephyr.plugin.core.api.monitoring.annotations.Monitor;

/**
 * Created by Marco Tamassia.
 */

// Off-policy TD() with a true online equivalence
public class GTDLambdaTrue implements  OnPolicyTD, GVF, EligibilityTraceAlgorithm {
    private static final long serialVersionUID = -6212873634659655856L;
    protected double gamma;
    final public double alpha_v;
    public final double alpha_w;
    protected double lambda_t;
    private double gamma_t;
    @Monitor(level = 4)
    final public PVector v;
    @Monitor(level = 4)
    protected final PVector w;
    private final Traces e;
    @Monitor(wrappers = { Squared.ID, Abs.ID })
    protected double delta_t;
    private double correction;
    private double rho_t;


    protected double v_t, v_tp1, v_old, gamma_tp1, lambda_tp1, rho_tm1;
    protected Traces e_d;
    protected Traces e_w;

    public GTDLambdaTrue(double alpha_v, double alpha_w, double gamma_t, double lambda_t, int nbFeatures,
        Traces prototype) {
        this.v_t = 0;
        this.v_tp1 = 0;
        this.v_old = 0;
        this.gamma_tp1 = 0;
        lambda_tp1 = 0;
        rho_tm1 = 0;
        this.alpha_v = alpha_v;
        this.alpha_w = alpha_w;
        this.gamma_t = gamma_t;
        this.lambda_t = lambda_t;
        this.e = prototype.newTraces(nbFeatures);
        this.e_d = prototype.newTraces(nbFeatures);
        this.e_w = prototype.newTraces(nbFeatures);
        this.v = new PVector(nbFeatures);
        this.w = new PVector(nbFeatures);
    }

    double initialize()
    {
        initialize();
        e_d.clear();
        e_w.clear();
        v_old = 0;
        rho_tm1 = 0;
        return 0.0;
    }

    @Override
    public void resetWeight(int index) {
        v.data[index] = 0;
        e.vect().setEntry(index, 0);
    }

    @Override
    public double update(RealVector x_t, RealVector x_tp1, double r_tp1) {
        return update(1, 1, x_t, x_tp1, r_tp1, gamma, 0, 0);
    }

    @Override
    public double update(double pi_t, double b_t, RealVector x_t, RealVector x_tp1, double r_tp1) {
        return update(pi_t, b_t, x_t, x_tp1, r_tp1, gamma, 0, 0);
    }

    @Override
    public double update(double pi_t, double b_t, RealVector phi_t, RealVector phi_tp1, double gamma_tp1,
					double lambda_tp1, double r_tp1, double z_tp1)
    {
        v_t = v.dotProduct(phi_t);
        v_tp1 = v.dotProduct(phi_tp1);
        delta_t = r_tp1 + (1.0 - gamma_tp1) * z_tp1 + gamma_tp1 * v_tp1 - v_t;

        // e
        rho_t = pi_t / b_t;
        e.update(gamma_t * lambda_t, phi_t, alpha_v * (1.0 - rho_t * gamma_t * lambda_t * e.vect().dotProduct(phi_t)));
        e.vect().mapMultiplyToSelf(rho_t);

        // e^{\Delta}
        e_d.update(gamma_t * lambda_t, phi_t);
        e_d.vect().mapMultiplyToSelf(rho_t);

        // e^w
        e_w.update(rho_tm1 * gamma_t * lambda_t, phi_t, alpha_w * (1.0 - rho_tm1 * gamma_t * lambda_t * e_w.vect().dotProduct(phi_t)));

        // v
        // part 1
        v.addToSelf((delta_t + v_t - v_old), e.vect());
        // part 2
        v.addToSelf(-alpha_v * rho_t * (v_t - v_old), phi_t);
        // part3
        v.addToSelf(-alpha_v * gamma_tp1 * (1.0 - lambda_tp1) * w.dotProduct(e_d.vect()), phi_tp1);

        // w
        // part 2
        w.addToSelf(-alpha_w * w.dotProduct(phi_t), phi_t);
        // part 1
        w.addToSelf(rho_t * delta_t, e_w.vect());

        gamma_t = gamma_tp1;
        lambda_t = lambda_tp1;
        rho_tm1 = rho_t;
        v_old = v_tp1;
        return delta_t;
    }

    void reset()
    {
        reset();
        e_d.clear();
        e_w.clear();
        v_old = 0;
        rho_tm1 = 0;
    }

    @Override
    public double predict(RealVector phi) {
        return v.dotProduct(phi);
    }

    @Override
    public PVector weights() {
        return v;
    }

    @Override
    public PVector secondaryWeights() {
        return w;
    }

    @Override
    public Traces traces() {
        return e;
    }

    @Override
    public double error() {
        return delta_t;
    }

    @Override
    public double prediction() {
        return v_t;
    }

}
