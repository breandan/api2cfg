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
