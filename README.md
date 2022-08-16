<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->

<!-- PROJECT LOGO -->
<br />
<div align="center">
    <img src="https://www.printablee.com/postpic/2009/12/printable-sign-language-numbers-chart_313730.jpg" alt="Logo" width="700" height="300">

  <h1 align="center">Sign Language Image Clustering</h1>

  <p align="center">
    A research exploring various methods of clustering images
</div>


<!-- ABOUT THE PROJECT -->
## About The Project

Communication is very crucial to human beings, as it enables us to express ourselves.  
We communicate through speech, gestures, body language, reading, writing, or through visual aids.  
speech is one of the most commonly used among them.  
Unfortunately, for the speaking and hearing impaired minority, there is a communication gap.  
Visual aids, or an interpreter, are used for communicating with them.  
However, these methods are rather cumbersome and expensive, and can't be used in an emergency.  
Sign Language chiefly uses manual communication to convey meaning.   
This involves simultaneously combining hand shapes, orientations, and movement of the hands, arms, or body to express the speaker's thoughts.
  
Sign Language consists of fingerspelling, which spells out words character by character, and word level association which involves hand gestures that convey the word's meaning.  
Fingerspelling is a vital tool in sign language, as it enables the communication of names, addresses, and other words that do not carry meaning in the word-level association.  
In spite of this, fingerspelling is not widely used as it is challenging to understand and difficult to use.  
Moreover, there is no universal sign language, and very few people know it, making it an inadequate communication alternative

<!-- GETTING STARTED -->
## Introduction

Sign Language (SL) is the primary language for the speaking and hearing impaired.  
Each country has its own SL that is different from other countries.  
Each sign in a language is represented with variant hand gestures, body movements, and facial expressions.
The Hebrew Sign Language is the communication method for Hebrew hearing-impaired people.  
Our goal is to improve the dynamic Hebrew Sign Language translation. 
A system for sign language recognition that translates fingerspelling can solve this problem.  
Various machine learning algorithms are used and their accuracies are recorded and compared in this report.  


## Algorithms used in this paper

The point of this research is to validate the most efficient algorithm to provide the highest-end image clustering technique.  
We have come across several relevant algorithms that we will explore.  
With each algorithm we will use a few feature extraction techniques, in order to test the efficiency of the algorithms properly.  
The algorithms we used for clustering are K-Means and Gaussian Mixture. 


## Data

The data we used for our research is in this OneDrive folder -    
https://1drv.ms/u/s!Aqmah9OMflvlgZgjmS9m1ZKQU_n22g?e=hfjwVl  
The compressed file contains 3 folders: test, train and validation.   
All images are images of the Hebrew Alphabet, post processing.  
To simplify the input images, a binary mask is applied, and the hand’s edges are highlighted.  
The binary mask consists of gray-scaling, blurring, and applying thresholding.   
We would like to thank Romansko/SignLanguageRecognition: Hebrew sign language real time recognition using CNN, Keras & OpenCV. (github.com) for making Hebrew SL data available.



<!-- USAGE EXAMPLES -->
## References
 * A Brief Introduction to Unsupervised Learning | by Aidan Wilson | Towards Data Science  
* Feature Extraction - MATLAB & Simulink (mathworks.com)  
 * What Is the Mean | How to Find It & Examples (scribbr.com)  
 * Understanding K-means Clustering in Machine Learning | by Education Ecosystem (LEDU) | Towards Data Science  
 * Gaussian Mixture Models (mit.edu)  
 * Gaussian Mixture Models for Clustering | by Vivienne DiFrancesco | Towards Data Science  
 * ResNet-50 convolutional neural network - MATLAB resnet50 (mathworks.com)  
 * Everything you need to know about VGG16 | by Great Learning | Medium  
 * Centroid — Google Arts & Culture  
 * Dimension_reduction (deepnote.com)  
 * Principal Component Analysis | LinkedIn    

<!-- ROADMAP -->
## Roadmap

- [x] Add Changelog
- [x] Add back to top links
- [ ] Add Additional Templates w/ Examples
- [ ] Add "components" document to easily copy & paste sections of the readme
- [ ] Multi-language Support
    - [ ] Chinese
    - [ ] Spanish

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
* [Malven's Grid Cheatsheet](https://grid.malven.co/)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)
* [React Icons](https://react-icons.github.io/react-icons/search)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
