# Initial public release checklist

- [ ] Review every file in the staged repository.
- [ ] Confirm `Vonkid` is the desired public author/copyright identity.
- [ ] Confirm MIT is the desired license.
- [ ] Run `python -m pytest -q` and `scholarhound --help`.
- [ ] Confirm the safety scan reports no private paths, credentials, or non-example email addresses.
- [ ] Create the public GitHub repository `Vonkid/ScholarHound` from this directory with a fresh history.
- [ ] Create and push tag `v0.1.0`.
- [ ] Create a GitHub release describing the alpha status and privacy boundary.
- [ ] Enable private vulnerability reporting and branch protection.
- [ ] Link the repository from `scholarhound.academy`.
- [ ] When ready, archive the tagged release in a DOI-granting repository for stronger scholarly provenance.

