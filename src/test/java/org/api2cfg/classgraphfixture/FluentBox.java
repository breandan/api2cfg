package org.api2cfg.classgraphfixture;

public class FluentBox<T extends Number> {
  private final T value;

  public FluentBox(T value) {
    this.value = value;
  }

  public T value() {
    return value;
  }

  public FluentBox<T> next(T next) {
    return new FluentBox<>(next);
  }

  public static FluentBox<Integer> create() {
    return new FluentBox<>(0);
  }

  public static FluentBox<Integer> pair(Integer first, Integer second) {
    return new FluentBox<>(second);
  }

  public FluentBox<T> join(T first, T second, T third) {
    return new FluentBox<>(third);
  }

  public FluentBox<T> tooWide(T first, T second, T third, T fourth) {
    return new FluentBox<>(fourth);
  }

  public <U extends Number> U echo(U other) {
    return other;
  }

  public Object choose(String value) {
    return value;
  }

  public String choose(Object value) {
    return value.toString();
  }
}
