package rltoys.experiments.parametersweep.tests.interfaces;

import java.util.HashSet;
import java.util.Set;

import org.junit.Assert;
import org.junit.Test;

import rltoys.experiments.parametersweep.parameters.FrozenParameters;
import rltoys.experiments.parametersweep.parameters.Parameters;


public class ParametersTest {
  static private String[] NoFlag = new String[] {};

  @Test
  public void testParametersEquals01NoFlags() {
    FrozenParameters p01 = createParameter01(NoFlag);
    FrozenParameters p01bis = createParameter01Bis(NoFlag);
    Assert.assertTrue(p01.equals(p01bis));
    Assert.assertEquals(p01.hashCode(), p01bis.hashCode());
    Set<FrozenParameters> set = new HashSet<FrozenParameters>();
    set.add(p01);
    set.add(p01bis);
    Assert.assertEquals(1, set.size());
    Assert.assertTrue(set.contains(createParameter01(NoFlag)));
    Assert.assertFalse(set.contains(createParameter02(NoFlag)));
    FrozenParameters p02 = createParameter02(NoFlag);
    Assert.assertFalse(p01.equals(p02));
    Assert.assertTrue(p01.hashCode() != p02.hashCode());
    set.add(p02);
    Assert.assertEquals(2, set.size());
  }

  @Test
  public void testParametersEquals01WithFlags() {
    FrozenParameters p01 = createParameter01(new String[] { "Flag01" });
    FrozenParameters p01bis = createParameter01Bis(new String[] { "Flag01" });
    Assert.assertTrue(p01.equals(p01bis));
    Assert.assertEquals(p01.hashCode(), p01bis.hashCode());
    p01bis = createParameter01Bis(new String[] { "Flag01Bis" });
    Assert.assertTrue(p01.equals(p01bis));
    Assert.assertFalse(p01.hashCode() != p01bis.hashCode());
  }

  private FrozenParameters toParameters(String[] flags, Object... objects) {
    Parameters parameters = new Parameters();
    for (int i = 0; i < objects.length / 2; i++)
      parameters.put((String) objects[i * 2], (double) (Integer) objects[i * 2 + 1]);
    for (String flag : flags)
      parameters.enableFlag(flag);
    return parameters.froze();
  }

  private FrozenParameters createParameter02(String[] flags) {
    return toParameters(flags, "Hello", 0, "Bye", 3);
  }

  private FrozenParameters createParameter01(String[] flags) {
    return toParameters(flags, "Hello", 0, "Bye", 2);
  }

  private FrozenParameters createParameter01Bis(String[] flags) {
    return toParameters(flags, "Bye", 2, "Hello", 0);
  }
}
