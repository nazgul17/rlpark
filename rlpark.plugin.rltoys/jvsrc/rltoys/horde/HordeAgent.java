package rltoys.horde;

import rltoys.algorithms.learning.control.Control;
import rltoys.algorithms.representations.Projector;
import rltoys.algorithms.representations.actions.Action;
import rltoys.environments.envio.RLAgent;
import rltoys.environments.envio.observations.TRStep;
import rltoys.math.vector.RealVector;
import zephyr.plugin.core.api.monitoring.annotations.Monitor;

public class HordeAgent implements RLAgent {
  private static final long serialVersionUID = -8430893512617299110L;

  @Monitor
  protected final Control control;
  @Monitor
  protected final Projector projector;
  protected RealVector x_t;
  @Monitor
  private final Horde horde;

  public HordeAgent(Control control, Projector projector, Horde horde) {
    this.control = control;
    this.projector = projector;
    this.horde = horde;
  }

  @Override
  public Action getAtp1(TRStep step) {
    if (step.isEpisodeStarting())
      x_t = null;
    RealVector x_tp1 = projector.project(step.o_tp1);
    Action a_tp1 = control.step(x_t, step.a_t, x_tp1, step.r_tp1);
    horde.update(step, x_t, step.a_t, x_tp1);
    x_t = x_tp1;
    return a_tp1;
  }

  public Control control() {
    return control;
  }

  public Projector projector() {
    return projector;
  }

  public Horde horde() {
    return horde;
  }
}
