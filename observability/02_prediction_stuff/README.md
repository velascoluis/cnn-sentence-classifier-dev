# Prediction rig engineering

(This is WIP)

The prediction rig is a component a bit overlooked when designing production scale ML platforms.
Other parts like components for supporting EdA or training tend to get more engineering focus.
But at the end of the days, the value generation process occur precisely at predicion phase and is also the external interface for investments on ML.
I think that inside the rig there should be at least the following components:


- **Feature transformer**: Even when having a feature store in place for decoupling features from data production systems, I think a feature transformer still has its place at inference time to apply low level specific operations to a potential reusable and more abstract feature. For online systems, latency requirements are critical.
- **Dispatcher**: The dispatcher objective is to route requests to a particular prediction endpoint. I believe that every single request should be subjected to an experiment, that’s why the dispatcher should be able to redirect the call to a particular or many live experiment(s), to the golden model or to both. Each request not subject to experimentation is an improvement opportunity lost.
- **Predict backbone**: The horsepower of the prediction rig resides on this component, hence from an engineering standpoint, it would be critical to design for classical non-functional requirements such as performance, scalability or fault tolerance ..
- **Cache layer**: Low latency key value store to quickly respond to re-entrant queries. It must implement the classical cache mechanisms (invalidations, key computes based on feature hashing, LRUs queues ..)
- **Golden promoter/de-promoter**: As A/B tests take place, we could potentially reach to a point where one of the live experiments is actually most performant that the current gold model, this component mission is to analyse metadata and particularly ground truth data in the feature store to suggest a replacement of the golden model with one of the experiments.
- **Model warmers**: Component to ensure cache and memory warm-ups when a cold star situation happen (e.g. new model promotion)
- **Explainer**: Component that implement model explainability logic (e.g. Anchors, CEM ..) and returns it for a given request
- **Metadata Store**: This component centralised all the metadata associated with the prediction phase (live experiments performance, prediction data stats …)

![Prediction rig](https://miro.medium.com/max/700/0*-B48knqCeg7LnBJB)

I am building notebooks to illustrate these concepts, main tools are KFServing, FEAST and some TF utilities:

- [X] Use of KFServing 
- [ ] Use of KFServing with data transformer
- [ ] Use of KFServing together with TF model warmer
- [X] Use of FEAST for online pred
- [ ] Use of FEAST for batch pred
- [X] Export a BQML model and deployment on KFServing