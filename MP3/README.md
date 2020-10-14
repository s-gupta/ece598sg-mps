## ECE598SG (Special Topics in Robot Learning)
### Programming Assignment 3
In this programming assignment you will implement basic behavior cloning for
discrete (`CartPole`) and continuous control tasks (`DoubleIntegrator` and
`PendulumBalance`). We have provided data from an expert. You have to use this
expert data to train policies to solve these tasks. 

1. **Behavior Cloning from States**. Using the expert data as supervision,
   implement behavior cloning for the 3 environments. For the discrete task,
   you can use a cross entropy loss, and for the continuous task you can use a
   mean-squared error loss. We have provided code for loading in the expert
   data (see `run_behavior_cloning.py`), and a simple feed-forward policy (see
   `policies.py`). You need to implement the training loop. You will measure
   performance of your policies, by running it in 100 validation episodes.
   1. **Implementation [15pts].** Get your implementation to work and solve the
      3 tasks. You may have to play around with hyper-parameters for traininggq
      the network. You can also tweak the policy architecture if you'd like.
      Report the average reward, and the metrics that your policy is able to
      achieve on the 3 environmets. You can get started by running the
      following command: `python run_behavior_cloning.py --logdir LOGDIR
      --env_name CartPole-v2`. Our implementation was able to solve the tasks
      within 5 minutes of training.
   2. **Data efficiency of learning [15pts].** Study the performance of your
      policy as a function of the number of expert demonstrations. You should
      vary the number of expert episodes on a log scale (so, keep halving them
      from however many there are in total). Make sure you train the network to
      convergence for each run. Plot the rewards and the metrics for the 3
      environments as a function of the number of expert demonstrations.
1. **[Extra Credit] Behavior Cloning from Images [5pts]**. We have also provided
     visual analogues of the three environments (with `Visual` pre-pended to
     their names). We have also provided corresponding expert demonstrations (
     you can generate them using `python generate_visual_data.py --env_name
     CartPole-v2`). Your task is to train policies that use renderings of the
     environment instead of the underlying state. Note that a single image will
     be insufficient to estimate the full state of the system, so you may have
     to use the last 2 frames as input to this policy. While we have provided a
     basic CNN policy for this, you may have to experiment with the
     architecture of the policy. Again, implement the training loop, get your
     policies to work, and report the performance of your policy. You can run
     these experiments by calling:`python run_behavior_cloning.py --logdir
     LOGDIR --env_name VisualCartPole-v2`. If you are running on a headless
     server, then you will need to use `xvfb` for generating the data and
     evaluating your policies.

#### Instructions
1. Assignment is due at 11:59:59PM on Wednesday October 28, 2020.
2. Please see
[policies](http://saurabhg.web.illinois.edu/teaching/ece598sg/fa2020/policies.html)
on [course
website](http://saurabhg.web.illinois.edu/teaching/ece598sg/fa2020/index.html).
3. Submission instructions:
   1. A single report for all questions in PDF format, to be submitted to
   gradescope (under assignment `MP3`).  Course code is `MZN3XY`. This report
   should contain all that you want us to look at. Please also tag your PDF
   with locations for each question in the gradescope interface.
   2. You also need to submit code for all questions in the form of a single .zip
   file. Please submit this under `MP3-code` on gradescope.
   3. We reserve the right to take off points for not following submission
   instructions.
4. You should be able to work on problem 1 even without a GPU, though you will
likely need a GPU for training visual policies over the extra-credit problem.
If you do happen to need a GPU, you can use GPUs on the [campus
cluster](http://saurabhg.web.illinois.edu/teaching/ece598sg/fa2020/compute.html),
and also through Google Colab. Please see course website for instructions on
how to use campus cluster. Instructions to use Google Colab can be found in
[MP1](../MP1).
5. Lastly, be careful not to work of a public fork of this repo. Make a *private*
clone to work on your assignment. 
