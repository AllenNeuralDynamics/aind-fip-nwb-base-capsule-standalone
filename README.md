# Fiber NWB Packaging Capsule


FiberPhotometry Capsule which appends to an NWB subject file. Adds FiberPhotometry information if present.

[Capsule link here](https://codeocean.allenneuraldynamics.org/capsule/3598308/tree)

[Pipeline link here](https://codeocean.allenneuraldynamics.org/capsule/7378248/tree)

NOTE: This capsule is in development with the file format standards for Fiber Photometry defined here: [Fiber Photometry Acquisition File Standards](https://github.com/AllenNeuralDynamics/aind-file-standards/blob/main/docs/file_formats/fip.md).

### Input and Output
The input to this capsule is the raw data (see file formats link for more details). The capsule will then output a NWB file with the raw fiber data added to the NWB. Under the `acquisition` field in the NWB, the following should be present as of now (note: this is subject to change depending on feedback for naming, what to store, etc.):
### 📑 TimeSeries
- `G_0`
- `G_1`
- `G_2`
- `G_3`
- `Iso_0`
- `Iso_1`
- `Iso_2`
- `Iso_3`
- `R_0`
- `R_1`
- `R_2`
- `R_3`

Where G (green), R (red), and Iso are the respective channels with 4-fiber connections 0-indexed. Each timeseries module has timestamps (on the HARP clock) and the data. See the file standards link above for more details on raw data acquisition.

### To test

You need a raw fiber data asset to test this capsule. 

The easiest way to do this is to go to the capsule link (the capsule IN the pipeline), and duplicate the capsule (linked to Git repo) when you want to edit it. Please make sure you rename your own personal capsule, and you use `git branch` to have any changes pushed to a new git branch.

### Creating a New Branch for contributing/testing

1. **Open Terminal or VSCode** 

2. **Create a new branch using `git checkout -b branch-name`** 

3. **Commit any edits**
   
4. **Push any edits to the git repo using `git push origin branch-name`**

5. **Create a PR to main using your prefered tools or the github website**

TODO: We should have MORE data to test this on– Ahad is working [on a ticket here](https://github.com/orgs/AllenNeuralDynamics/projects/75/views/1?filterQuery=ahad&pane=issue&itemId=81950029&issue=AllenNeuralDynamics%7Caind-physio-arch%7C278)

Note: When you are doing test runs on VS Code on Code Ocean, you need to **delete** whatever is in results folder every time you debug. 

Here is a combination that works: 
 
**Fiber data asset**

Data Name: behavior_700708_2024-06-13_09-06-26

Data Asset ID: 09adf0cb-dccc-48e5-8c5f-323a091895a6


# Contributing

## Issues and Feature Requests
Please create an issue if you have any questions, feature requests and bug reports you'd like the team to know about.

## For any non-trivial changes, please create a branch and use a Pull Request to main 
### Editing the Capsule
The easiest way to do this is to go to the capsule link (the capsule IN the pipeline), and duplicate the capsule (linked to Git repo) when you want to edit it. Please make sure you rename your own personal capsule, and you use `git branch` to have any changes pushed to a new git branch. You'll want to use either the terminal or VSCode to do so and instructions are provided below if needed.

### Creating a New Branch for contributing/testing

1. **Open Terminal or VSCode** 

2. **Create a new branch using `git checkout -b branch-name`** 

3. **Commit any edits**
   
4. **Push any edits to the git repo using `git push origin branch-name`**

5. **Create a PR to main using your prefered tools or the github website**


