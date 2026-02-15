# pdffigures2 setup

This project uses **pdffigures2** to extract figure images + metadata from PDFs.

After building the jar, set:
```bash
export PDFFIGURES2_JAR=/path/to/pdffigures2/target/scala-2.12/pdffigures2-assembly-*.jar
```
All venue pipelines read `PDFFIGURES2_JAR` at runtime.

## Local install (no admin access)

### 1) Java 8
```bash
mkdir -p ~/local/java
cd ~/local/java

wget -O temurin8.tar.gz \
  https://github.com/adoptium/temurin8-binaries/releases/download/jdk8u392-b08/OpenJDK8U-jdk_x64_linux_hotspot_8u392b08.tar.gz

tar -xzf temurin8.tar.gz
rm temurin8.tar.gz

# Find extracted folder name (e.g., jdk8u392-b08)
ls
```

### 2) sbt
```bash
mkdir -p ~/local/sbt
cd ~/local/sbt

wget -O sbt.tgz https://github.com/sbt/sbt/releases/download/v1.9.7/sbt-1.9.7.tgz

tar -xzf sbt.tgz
rm sbt.tgz
```

### 3) Export PATH/JAVA_HOME
Replace `jdk8u392-b08` with the folder you extracted.
```bash
export JAVA_HOME="$HOME/local/java/jdk8u392-b08"
export PATH="$JAVA_HOME/bin:$HOME/local/sbt/sbt/bin:$PATH"

java -version
sbt --version
```

## Build pdffigures2
```bash
cd ~
git clone https://github.com/allenai/pdffigures2.git
cd pdffigures2

sbt assembly

# Find the jar
ls -la target/scala-2.12/*.jar

# Set env var for the pipelines
export PDFFIGURES2_JAR="$PWD/target/scala-2.12/pdffigures2-assembly-*.jar"
```