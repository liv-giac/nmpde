export mkPrefix=/u/sw/
source ${mkPrefix}/etc/profile

module load gcc-glibc/11 dealii
cd /u/desantis/nmpde/lab-05/src
make