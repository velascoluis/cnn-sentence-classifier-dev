# Prediction rig engineering

(WIP)
The prediction rig is a component a bit overlooked when designing production scale ML platforms.
Other parts like components for supporting EdA or training tend to get more engineering focus.
But at the end of the days, the value generation process occur precisely at predicion phase and is also the external interface for investments on ML.
I think that inside the rig there should be at least the following components:

- A prediction endpoint, of course, with capabilities for:
    - Running low latency predictions
    - Scalability, fault tolerance ..
    - Advance loggig capabilities particulary important for ground truth checking
    - A/B canary rollout capabilities
    - Explanability
- A feature transformer:
    - In online models there tend to be a gap between the raw data and the data used for running the prediction.
- A prediction transformer:
    - blah
- A model warmer:
    - blah
- A Feature Store for decoupling data producers and features
    - Online
    - Batch
    - blah blah 
- A model (de)promoter

..insert diagram of a full prediction rig

This NB currenty shows:

- [X] Use of KFServing 
- [ ] Use of KFServing with data transformer
- [ ] Use of KFServing together with TF model warmer
- [X] Use of FEAST for online pred
- [ ] Use of FEAST for batch pred