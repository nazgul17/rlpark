package rltoys.environments.envio.agents;

import rltoys.algorithms.representations.actions.Action;
import rltoys.environments.envio.RLAgent;
import rltoys.environments.envio.control.ControlLearner;
import rltoys.environments.envio.observations.TRStep;
import rltoys.math.vector.RealVector;
import rltoys.math.vector.implementations.PVector;
import zephyr.plugin.core.api.monitoring.annotations.Monitor;

public class LearnerAgent implements RLAgent {
  private static final long serialVersionUID = -8694734303900854141L;
  @Monitor
  protected final ControlLearner control;
  protected RealVector x_t;

  public LearnerAgent(ControlLearner control) {
    this.control = control;
  }

  @Override
  public Action getAtp1(TRStep step) {
    if (step.isEpisodeStarting())
      x_t = null;
    PVector x_tp1 = new PVector(step.o_tp1);
    Action a_tp1 = control.step(x_t, step.a_t, x_tp1, step.r_tp1);
    x_t = x_tp1;
    return a_tp1;
  }

  public ControlLearner control() {
    return control;
  }
}
