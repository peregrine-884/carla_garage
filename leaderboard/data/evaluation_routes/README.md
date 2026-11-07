# Town05 evaluation routes

Closed-loop evaluation routes used by VAD / VADv2 / TransFuser-style papers.

| File | Routes | Source | SHA-256 |
|---|---:|---|---|
| `routes_town05_short.xml` | 32 | autonomousvision/transfuser `cvpr2021`: `leaderboard/data/validation_routes/routes_town05_short.xml` | `1f2eb95eedaf505a454b9b02293d7f13480736b6cb0deb95be50d29202af6cf2` |
| `routes_town05_long.xml` | 10 | autonomousvision/transfuser `cvpr2021`: `leaderboard/data/evaluation_routes/routes_town05_long.xml` | `de85c72b0fd6ba5a65a7e97b691dc26fc24ad407abd690fa59ef4c454ece3ee7` |

Both files are copied unchanged (TransFuser is MIT licensed). Use them with
`leaderboard/data/scenarios/town05_all_scenarios.json`, which is identical to the
file in the same TransFuser branch.
