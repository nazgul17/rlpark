package rltoys.experiments.parametersweep.offpolicy;

import rltoys.algorithms.representations.Projector;
import rltoys.environments.envio.RLAgent;
import rltoys.environments.envio.Runner;
import rltoys.environments.envio.problems.RLProblem;
import rltoys.experiments.ExperimentCounter;
import rltoys.experiments.parametersweep.offpolicy.evaluation.OffPolicyEvaluation;
import rltoys.experiments.parametersweep.parameters.Parameters;
import rltoys.experiments.parametersweep.reinforcementlearning.OffPolicyAgentFactory;
import rltoys.experiments.parametersweep.reinforcementlearning.ProblemFactory;
import rltoys.experiments.parametersweep.reinforcementlearning.ProjectorFactory;
import rltoys.experiments.parametersweep.reinforcementlearning.ReinforcementLearningContext;

public abstract class AbstractContextOffPolicy implements ReinforcementLearningContext {
  private static final long serialVersionUID = -6212106048889219995L;
  private final OffPolicyAgentFactory agentFactory;
  protected final ProblemFactory environmentFactory;
  protected final OffPolicyEvaluation evaluation;
  protected final ProjectorFactory projectorFactory;

  public AbstractContextOffPolicy(ProblemFactory environmentFactory, ProjectorFactory projectorFactory,
      OffPolicyAgentFactory agentFactory, OffPolicyEvaluation evaluation) {
    this.evaluation = evaluation;
    this.projectorFactory = projectorFactory;
    this.environmentFactory = environmentFactory;
    this.agentFactory = agentFactory;
  }

  @Override
  public Runner createRunner(int counter, Parameters parameters) {
    RLProblem problem = environmentFactory.createEnvironment(ExperimentCounter.newRandom(counter));
    Projector projector = projectorFactory.createProjector(problem);
    RLAgent agent = agentFactory.createAgent(problem, projector, parameters, ExperimentCounter.newRandom(counter));
    int nbEpisode = parameters.nbEpisode();
    int maxEpisodeTimeSteps = parameters.maxEpisodeTimeSteps();
    return new Runner(problem, agent, nbEpisode, maxEpisodeTimeSteps);
  }

  @Override
  public String fileName() {
    return ExperimentCounter.DefaultFileName;
  }

  @Override
  public String folderPath() {
    return environmentFactory.label() + "/" + agentFactory.label();
  }

  public OffPolicyAgentFactory agentFactory() {
    return agentFactory;
  }

  public ProblemFactory problemFactory() {
    return environmentFactory;
  }

  public Parameters contextParameters() {
    Parameters parameters = new Parameters();
    environmentFactory.setExperimentParameters(parameters);
    parameters.enableFlag(agentFactory.label());
    parameters.enableFlag(environmentFactory.label());
    return parameters;
  }
}
