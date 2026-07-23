plugins {
  kotlin("jvm") version "2.4.0"
  application
}

group = "org.lib2cfg"
version = "1.0-SNAPSHOT"

repositories {
  mavenCentral()
}

dependencies {
  implementation(kotlin("reflect"))
  testImplementation(kotlin("test"))
  implementation("io.github.classgraph:classgraph:4.8.184")
}

kotlin {
  jvmToolchain(21)
}

tasks.test {
  useJUnitPlatform()
}

application {
  mainClass.set("org.lib2cfg.MainKt")
}

tasks.register<JavaExec>("runClassGraph") {
  group = "application"
  description = "Generate a grammar with the ClassGraph-backed implementation"
  classpath = sourceSets["main"].runtimeClasspath
  mainClass.set("org.lib2cfg.ClassGraphKt")
}
