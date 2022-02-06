To run the following code setup Jekyll on your computer

Refer: https://jekyllrb.com/docs/installation/ubuntu/
Refer: https://jekyllrb.com/docs/step-by-step/01-setup/

The above links should follow the same procedure as shown below:
(Please recheck below commands with above links to stay updated)

Jekyll + Ruby Installation:

1. sudo apt-get install ruby-full build-essential zlib1g-dev
2. echo '# Install Ruby Gems to ~/gems' >> ~/.bashrc
3. echo 'export GEM_HOME="$HOME/gems"' >> ~/.bashrc
4. echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.bashrc
5. source ~/.bashrc

6. gem install jekyll bundler

7. bundle init

For example jekyll project do

8. gem "jekyll"

Or clone the repo in local machine and run the following command within the working directory:

bundle exec jekyll serve
