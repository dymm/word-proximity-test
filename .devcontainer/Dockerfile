FROM mcr.microsoft.com/devcontainers/java:21

# Install PostgreSQL client tools and Maven (plus helpers)
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends postgresql-client maven wget unzip ca-certificates \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install Gradle
ARG GRADLE_VERSION=8.5
RUN wget https://services.gradle.org/distributions/gradle-${GRADLE_VERSION}-bin.zip -P /tmp \
    && unzip -d /opt/gradle /tmp/gradle-${GRADLE_VERSION}-bin.zip \
    && ln -s /opt/gradle/gradle-${GRADLE_VERSION}/bin/gradle /usr/bin/gradle \
    && rm /tmp/gradle-${GRADLE_VERSION}-bin.zip

# Verify installations
RUN java -version && mvn -version && gradle -version
