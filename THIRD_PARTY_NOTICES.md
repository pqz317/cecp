# Third-party notices

## jax_pbt (MIT)
The core of `cecp` (environments API, models, PPO trainer, controllers) is derived from
[Social-RL/jax_pbt](https://github.com/Social-RL/jax_pbt), Copyright (c) 2024 Social RL, MIT License.
Its copyright line is retained in `LICENSE`.

## JaxMARL Overcooked (Apache-2.0) via crossEnvCooperation (MIT)
`cecp/envs/overcooked/_jaxmarl/` contains a modified copy of the Overcooked environment, layouts and grid
renderer from:
- [JaxMARL](https://github.com/FLAIROx/JaxMARL), Copyright 2023 FLAIR, Apache License 2.0
  (`cecp/envs/overcooked/_jaxmarl/LICENSE-APACHE`), as extended by
- [crossEnvCooperation](https://github.com/pqz317/crossEnvCooperation) (Jha et al., 2025, procedurally
  generated 9x9 layouts), Copyright (c) 2025 Kunal Jha, MIT License
  (`cecp/envs/overcooked/_jaxmarl/LICENSE-MIT-crossEnvCooperation`).

Each vendored file carries a header noting its origin. Modifications for `cecp`: imports made relative,
debugger imports removed, and window/animation/CLI rendering code removed (only the array renderer is kept).
