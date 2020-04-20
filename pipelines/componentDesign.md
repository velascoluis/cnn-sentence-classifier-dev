# Notes on component design
@to-do  
Reusable components are a major feature of kubeflow pipelines, a component represents an abstraction of a reusable step in a ML workflow pipeline. For instance:
- Data preparation component
- Model deployer component
- Metadata logging component 
- ...

As the [documentation cites](https://www.kubeflow.org/docs/pipelines/sdk/build-component/)  a component consists of an interface (inputs/outputs), the implementation (a Docker container image and command-line arguments) and metadata (name, description).

---
Now, in order to maximize effiency in ML development there are a few considerations when designing a reusable component:

**Granularity:**  
- How "big" should be the component packaged logic? There is no definitive answer to this, but I can think of some guidelines and trade-offs to consider.  
 At the very least for initial deployments I think we should partition the code/components following the build-train-deploy pattern.
 The build phase is probably the one we can sub-partition more, in particular the data preparation steps and feature generation.
 On the other hand, it will add little value to encapsulate simple logic into to components, for instance, loading data from GCS or performing some simple data cleansing.   
 
 SIMPLE DEPLOYMENT
 
 **Components proposed for the build phase***
 **Components proposed for the train phase***
 **Components proposed for the deploy phase***
 
 
 
 CICD PIPELINES
  - LaunchHPLRO
  - LaunchNASLRO
  - CheckModelCandidates
  - PromotoToProd
  - PromotoToProdCanary
  - ChangeCanaryConfig
  - SignatureGen
  -ExplainModel
 ***Orchestrated experiment***
 ***Decomissions components***
 ***Ongoing checks***
 - Evaluators
    - FeatureQuality
    - Schema change
    - Skew 
    - Drfit 
    - Accuracy 
 - GoldenRetrainerLauncher (distributed, GPU options ...)
 - Metrics collector

 
   
@to-do  
**Interface design** 
- Input/output parameters using input/outputPath vs input/outputValue vs external storage.   
**Observability**  
- Dataset common metrics  
- Training phase commom metrics  
- Deployment metrics  
- Inference commom metrics
- SLO list  

**Fit language/purpose**   
 @to-do  
**Container annotations (e.g. hardware acceleration)**    
@to-do  
**Component testing**   
 @to-do  
**Version control**   