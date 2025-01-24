## Assignment 4: Linear and Nonlinear equalization
This repository contains the report and the Python scripts compiled to address the final hand-in requirements for the DTU Winter 2025 course 34242: Machine learning techniques for design and optimization of communication systems.

## Project Contributers
<ul>
  <li>Enrico Leonardi (student - 222721)</li>
</ul>

## Abstract:
The assignment includes 4 exercises:
<ol type="I">
  <li>The objective of this exercise is to implement linear discrete-time communication system model, and show that the linear adaptive linear equalizer is effective in combating distortions induced by the channel.</li>
  <li>In this exercise, the goal is to demonstrate the capabilities of a single-layer neural network, trained by gradient descent (backpropagation), to model various functions.</li>
  <li>The objective of this exercise is to implement the nonlinear discrete-time communication channel with the nonlinear equalizer. You will demonstrate that the nonlinear equalizer, implemented as a multiple input single layer neural network, is effective in combating distortions induced by the nonlinear channel.</li>
  <li>Finally, implement a convolutional nonlinear equalizer using the techniques implemented so far, and evaluate its performance on the nonlinear channel specified by the parameters from the previous exercise.</li>
</ol>

## Data
The employed scripts have been primarily developed by me. You can check how I addressed each one of the exercises by glimpsing into each exercise script. A small PDF report is also available for showcasing and discussing the results.

## Goal
The report has been the final of 4 mandatory hand-ins required to succeed the course. The objective of the course has been to introduce basic machine learning techniques that can be applied for designing fiber-optic and as well as wireless communication systems. The course provided us students with knowledge of how to: 1) build data driven models of from input/output data. This will include models of amplifiers, transmission channels, transmitter-receivers, 2) design pre- and –post nonlinear equalizers based on multi-layer neural-network. 3) optimal filtering and recover useful information from noisy observations and 4) perform system performance prediction and optimization of communication systems and networks.

## Usage
The suggested means of usage of the provided material is by reading through the PDF report. In aid of the concepts, the Python scripts can be checked to understand how to manually implement adaptive equalizers and single-layer neural networks based on gradient descent.

## Conclusion
Machine learning is becoming an indispensable tool for communication engineers. However, as the field of machine learning is broad, identifying a set of relevant tools for solving a specific set of problems may be challenging. <br/><br/> Specifically for this assignment, linear and nonlinear adaptive equalizer models have been implemented. Moreover, it has been successfully shown how a signal - distorted by such modeled systems - can be restored by resorting on machine learning algorithms (e.g. gradient descent).
