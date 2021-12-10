The docker image works:

    docker run -it --entrypoint bash -w /app/TikZ -v"$PWD":/transfer-volume ellisk/graphics:muse


    apt-get update
    apt-get upgrade
    apt-get install texlive-latex-extra imagemagick vim

Find the policy.xml with find / -name "policy.xml"
(it was `/etc/ImageMagick-6/policy.xml` for me)
and change

    vi /etc/ImageMagick-6/policy.xml

    <policy domain="resource" name="disk" value="1GiB"/>

to

    <policy domain="resource" name="disk" value="8GiB"/>


I'm not sure if this was actually necessary, but I applied the changes
at
https://github.com/ianhuang0630/TikZ/commit/21ecc4e9beb800e15f0c6c6f91f5ab5254187852
before I upped the limit


Then you can synthesize data.  You need to do more than 100 or the

    python makeSyntheticData.py 101

Then train the proposal distribution

    python recognitionModel.py train --noisy  --attention 16 -n 101



Then you can exit from the container and run

    docker ps -a
    docker commit 8dfc732ba0b7 ellisk-graphics-updated

That will save an image with all your changes, so from there on out
you can run

    docker run -it --entrypoint bash -w /app/TikZ -v"$PWD":/transfer-volume ellisk-graphics-updated


I had to change the BatchIterator down to a batchsize of 1, otherwise,
it would run out of memory and fail in the docker container.
