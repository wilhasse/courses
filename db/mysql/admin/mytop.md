# Introduction

Install mytop on Debian 12


# Install

mytop

```bash
sudo cpan App::mytop
wget https://raw.githubusercontent.com/jzawodn/mytop/master/mytop
sudo chmod +x mytop
sudo mv mytop /usr/local/bin/
```

packages

```bash
sudo apt-get install libdbi-perl
sudo apt-get install libmysqlclient-dev
sudo apt-get install libdbd-mysql-perl
sudo apt-get install libterm-readkey-perl
```

create ~/.mytop

```ini
user=your_username
pass=your_password
host=hostname
db=database_name
```

correct:
Use of uninitialized value $host in substitution (s///) at /usr/local/bin/mytop line 958.

edit /usr/local/bin/mytop

```perl
# from
            my $host = gethostbyaddr(inet_aton($thread->{Host}), AF_INET);
            $host =~ s/^([^.]+).*/$1/;
# to
            my $host = $thread->{Host} || 'unknown';
            $host =~ s/^([^.]+).*/$1/;            
```            