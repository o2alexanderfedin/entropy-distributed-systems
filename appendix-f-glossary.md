# Appendix F: Glossary

**Author**: Analysis by AI Hive®  
**Date**: August 27, 2025  
**Version**: 1.0 - Technical Glossary  
**Context**: Definitions for Secured by Entropy P2P Cloud Framework  
**Main Document**: [Secured by Entropy P2P Cloud Academic Paper](./Secured_by_Entropy_P2P_Cloud_2025-08-25.md)

---

## Core Concepts

**Argon2id**  
Memory-hard key derivation function resistant to GPU/ASIC attacks. Uses 1GB memory requirement to prevent brute-force attacks on password-derived keys.

**Bootstrap Nodes**  
Initial entry points for joining a DHT network. Provide peer discovery and routing table initialization for new nodes.

**Byzantine Fault Tolerance**  
System property enabling correct operation despite arbitrary failures or malicious behavior from up to 1/3 of network participants.

**Constant-Time Operations**  
Cryptographic implementations where execution time is independent of input values, preventing timing-based side-channel attacks.

**Device-Bound Identity**  
Cryptographic identity tied to specific hardware through TPM, secure enclave, or hardware fingerprinting. Keys cannot be exported or transferred.

**DHT (Distributed Hash Table)**  
Decentralized data structure providing key-value storage across distributed nodes without central coordination. Uses consistent hashing for load distribution.

**ECDLP (Elliptic Curve Discrete Logarithm Problem)**  
Mathematical hard problem underlying elliptic curve cryptography. Breaking ECDLP would compromise Ed25519/X25519 security.

**Eclipse Attack**  
Network-level attack where adversary controls all connections to a target node, isolating it from the honest network and enabling manipulation.

**Ed25519**  
High-performance elliptic curve signature scheme using Curve25519. Provides 128-bit security level with fast verification.

**Entropy-Native**  
System design principle where entropy (randomness) is a first-class component integrated throughout the architecture, not added as an afterthought.

**Forward Secrecy**  
Cryptographic property ensuring past communications remain secure even if long-term keys are compromised. Achieved through ephemeral key generation.

## Cryptographic Primitives

**HKDF (HMAC-based Key Derivation Function)**  
Cryptographic function for expanding a short input key into multiple longer keys. Uses SHA-3 for entropy extraction and expansion.

**K-Bucket**  
Routing table data structure in Kademlia DHT containing up to k peer entries sorted by XOR distance from local node ID.

**Kademlia**  
Distributed hash table protocol using XOR distance metric for routing. Provides O(log n) lookup complexity with self-healing properties.

**Key Derivation Function (KDF)**  
Cryptographic function deriving one or more secret keys from a master secret using additional parameters like salt and iteration count.

**Mesh Networking**  
Network topology where nodes connect directly, dynamically, and non-hierarchically. Enables multi-hop communication and resilient routing.

**Min-Entropy**  
Worst-case measure of randomness in a distribution. H∞(X) = -log₂(max P(X=x)) represents minimum unpredictability.

**Multi-Hop Routing**  
Network communication technique where data traverses multiple intermediate nodes to reach destination. Essential for mesh networks and DHTs.

## Network Security

**Network Partitioning**  
Scenario where network splits into disconnected components due to failures, attacks, or connectivity issues. Systems must remain functional during partitions.

**Perfect Forward Secrecy**  
Strongest form of forward secrecy where each session uses unique ephemeral keys that are never reused or stored.

**Proof-of-Work**  
Consensus mechanism requiring computational effort to participate. In our system, used for Sybil resistance rather than consensus.

**Randomness Extractor**  
Cryptographic function converting weak randomness sources into nearly uniform random bits. SHA-3 serves this role in our entropy mixing.

**SHA-3**  
NIST-standardized cryptographic hash function based on Keccak. Provides collision resistance and serves as randomness extractor.

**Side-Channel Attack**  
Cryptographic attack exploiting physical implementation characteristics (timing, power consumption, electromagnetic emanations) rather than algorithmic weaknesses.

**Store-and-Forward**  
Communication technique where intermediate nodes temporarily store messages for later delivery when destination becomes reachable.

**Sybil Attack**  
Attack where adversary creates multiple fake identities to gain disproportionate influence in peer-to-peer systems.

## System Architecture

**TPM (Trusted Platform Module)**  
Hardware security chip providing secure key storage, random number generation, and cryptographic operations isolated from main processor.

**VRF (Verifiable Random Function)**  
Cryptographic primitive providing publicly verifiable randomness. Enables provably random selection while preventing manipulation.

**WebAssembly (WASM)**  
Portable binary instruction format enabling near-native performance in sandboxed environments. Provides security isolation for untrusted code execution.

**X25519**  
Elliptic curve Diffie-Hellman key agreement protocol using Curve25519. Enables secure key exchange over insecure channels.

**XOR Distance Metric**  
Distance function used in Kademlia DHT where distance between two node IDs is their bitwise XOR. Creates structured overlay network topology.

## Attack Vectors and Defenses

**ASIC Resistance**  
Property of algorithms designed to prevent specialized hardware advantages. Achieved through memory-hard functions like Argon2id.

**Birthday Attack**  
Cryptographic attack exploiting probability theory to find hash collisions faster than brute force. Relevant for hash function security analysis.

**Brute Force Attack**  
Attack attempting all possible keys or passwords. Defeated by sufficient key length and computational hardness (memory requirements).

**Chosen Plaintext Attack**  
Cryptanalytic attack where adversary can obtain ciphertext for arbitrary plaintext. Modern encryption schemes must resist such attacks.

**Denial of Service (DoS)**  
Attack preventing legitimate users from accessing system resources. Mitigated through rate limiting and proof-of-work requirements.

**Key Exhaustion Attack**  
Attack attempting to exhaust cryptographic key space through systematic enumeration. Prevented by sufficient key entropy and length.

**Man-in-the-Middle Attack**  
Attack where adversary intercepts and potentially modifies communications between parties. Prevented by end-to-end encryption and authentication.

**Replay Attack**  
Attack replaying previously captured valid data transmission. Prevented by timestamps, nonces, and sequence numbers.

**Traffic Analysis**  
Attack inferring information from communication patterns rather than content. Mitigated through constant-rate communication and padding.

## Performance and Scalability

**Asymptotic Complexity**  
Mathematical notation describing algorithm performance scaling. Our DHT provides O(log n) lookup complexity.

**Load Balancing**  
Technique distributing work evenly across system components. DHT hash functions provide natural load distribution.

**Network Churn**  
Rate of nodes joining and leaving the network. High churn challenges system stability and requires robust recovery mechanisms.

**Peer Discovery**  
Process of finding and connecting to other network participants. Uses entropy-enhanced mechanisms to prevent targeted attacks.

**Scalability**  
System property enabling performance maintenance as size increases. Our architecture scales logarithmically with node count.

**Throughput**  
Measure of system processing capacity, typically operations or bytes per second. Optimized through parallel processing and efficient algorithms.

---

## Mathematical Notation

**H(X)** - Shannon entropy of random variable X  
**H∞(X)** - Min-entropy of random variable X  
**P(event)** - Probability of event occurrence  
**O(f(n))** - Big-O notation for algorithmic complexity  
**⊕** - XOR (exclusive or) operation  
**||** - Concatenation operation  
**λ** - Security parameter (typically 256 bits)  
**k** - Threshold parameter for consensus/routing  
**n** - Total number of network nodes  
**m** - Number of malicious nodes  

---

## Acronyms

**AES** - Advanced Encryption Standard  
**API** - Application Programming Interface  
**CPU** - Central Processing Unit  
**DDoS** - Distributed Denial of Service  
**GCM** - Galois/Counter Mode  
**GPU** - Graphics Processing Unit  
**HMAC** - Hash-based Message Authentication Code  
**HSM** - Hardware Security Module  
**IoT** - Internet of Things  
**NIST** - National Institute of Standards and Technology  
**P2P** - Peer-to-Peer  
**PKI** - Public Key Infrastructure  
**RNG** - Random Number Generator  
**TLS** - Transport Layer Security  
**WASM** - WebAssembly  

---

*This glossary provides comprehensive definitions for technical terms used throughout the entropy-native P2P framework documentation.*