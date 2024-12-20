# Key Libraries Used in PolarDB-X SQL Project

This document outlines the most important libraries and dependencies used in the PolarDB-X SQL project, based on comprehensive analysis of the project's module structure and dependencies.

## Core Database Libraries
- **MySQL Connector** (version 5.1.49)
  - Official JDBC driver for MySQL connectivity

- **Apache Calcite**
  - Core SQL parsing and optimization framework
  - Extended with custom PolarDB-X optimizations

- **Druid** (version 1.2.8)
  - High-performance database connection pooling
  - Database monitoring capabilities

## Distributed Computing & MPP
- **Airlift Framework** (version 206)
  - Comprehensive distributed computing framework
  - Components include:
    - Bootstrap and configuration
    - Discovery and node management
    - HTTP server and JAX-RS support
    - Stats and monitoring
    - JSON processing

- **JGraphT** (version 0.9.0)
  - Graph theory library for topology management
  - Used in distributed query planning

- **Google Guice** (version 5.1.0)
  - Dependency injection framework
  - Component lifecycle management

## Network & Communication
- **Netty** (version 4.1.44.Final)
  - High-performance network application framework
  - Asynchronous event-driven networking

- **Protocol Buffers** (version 3.11.1)
  - Google's data serialization format

- **gRPC** (version 1.30.0)
  - High-performance RPC framework

- **Java-IPv6** (version 0.17)
  - IPv6 protocol support
  - Network address management

## Data Processing & Analytics
- **Apache Arrow** (version 11.0.0)
  - In-memory columnar data format
  - High-performance data processing

- **Apache ORC** (version 1.6.9)
  - Optimized row columnar file format
  - Efficient data storage

- **Apache Hadoop** (version 3.2.2)
  - Distributed storage support
  - Big data processing capabilities

## Optimization & Performance
- **OjAlgo** (version 43.0)
  - Mathematics and linear programming library
  - Query optimization support

- **Janino** (version 3.1.9)
  - Java compiler for runtime code optimization
  - Dynamic code generation

- **Stream Analytics** (version 2.9.5)
  - Stream processing library

- **Caffeine** (version 2.9.3)
  - High-performance caching library
  - Memory management

- **RoaringBitmap** (version 1.2.1)
  - Compressed bitmap index implementation
  - Efficient set operations

## System & Native Integration
- **JNA** (Java Native Access) (version 5.9.0)
  - Native code access
  - System-level integration

- **SIGAR** (version 1.6.4)
  - System information gathering
  - Resource monitoring

## Utility Libraries
- **Google Guava** (version 27.0.1-jre)
  - Core Java utilities and helper classes

- **Apache Commons**
  - commons-lang3 (version 3.8.1): Core Java language utilities
  - commons-io (version 2.4): IO operation utilities
  - commons-codec (version 1.11): Encoding/decoding utilities

## JSON & Data Format Processing
- **Fastjson** (version 1.2.83)
  - High-performance JSON processing
  - Alibaba's JSON library

- **Jackson** (version 2.13.1)
  - Robust JSON processing framework
  - Multiple data format support

- **SnakeYAML** (version 1.21)
  - YAML configuration processing

## Testing & Validation
- **Google Truth** (version 1.0)
  - Fluent testing assertions

- **EqualsVerifier** (version 3.16)
  - Testing equals/hashCode implementations

- **Mockito** (version 3.12.4)
  - Mocking framework

## Logging & Monitoring
- **SLF4J** (version 1.7.21)
  - Logging facade
  - Abstract logging framework

- **Logback** (version 1.2.3)
  - Logging implementation
  - Production-ready logging

- **Metrics** (version 4.2.18)
  - Dropwizard metrics
  - Application monitoring

## Project Overview
PolarDB-X SQL is a sophisticated distributed SQL engine that combines these libraries to create a highly scalable and performant database solution.  

The architecture leverages Apache Calcite for SQL processing, Airlift for distributed computing, and modern networking libraries like Netty and gRPC for component communication.  

The integration of mathematical optimization (OjAlgo), native system access (JNA, SIGAR), and comprehensive testing frameworks ensures robust performance, reliability, and maintainability.