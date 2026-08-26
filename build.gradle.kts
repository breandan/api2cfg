plugins {
  kotlin("jvm") version "2.4.0"
  application
}

group = "org.api2cfg"
version = "1.0-SNAPSHOT"

repositories { mavenCentral() }

dependencies {
  implementation(kotlin("reflect"))
  implementation("com.fasterxml.jackson.core:jackson-core:2.18.3")
  testImplementation(kotlin("test"))
  implementation("io.github.classgraph:classgraph:4.8.184")
}

kotlin { jvmToolchain(21) }

tasks.test { useJUnitPlatform() }

application { mainClass.set("org.api2cfg.MainKt") }

tasks.register<JavaExec>("runClassGraph") {
  group = "application"
  description = "Generate a grammar with the ClassGraph-backed implementation"
  classpath = sourceSets["main"].runtimeClasspath
  mainClass.set("org.api2cfg.ClassGraphKt")
}

tasks.register<JavaExec>("runCpp26") {
  group = "application"
  description = "Generate a C++26 standard-library statement grammar"
  classpath = sourceSets["main"].runtimeClasspath
  mainClass.set("org.api2cfg.cpp26.CliKt")
}