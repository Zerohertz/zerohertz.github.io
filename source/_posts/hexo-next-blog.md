---
title: Update Hexo NexT Blog
date: 2022-12-06 21:51:11
categories:
- Etc.
tags:
- GitHub
- Hexo
---
# Update npm

`node.js`와 `npm`을 업데이트하기 전 삭제를 해야하는 경우 아래의 글을 참고하면 됩니다.

## MacOS

[Uninstall homebrew](https://ddoongmause.blogspot.com/2021/02/brew.html)
[Uninstall node & npm](https://clolee.tistory.com/85)

<img width="762" alt="npm error" src="/images/hexo-next-blog/205918079-9c278225-b2a5-4fdd-831b-f006bdadf55c.png">

~~~applescript
brew install node
brew install yarn --ignore-dependencies
yarn set version stable
~~~

<img width="762" alt="Reinstalled node & npm" src="/images/hexo-next-blog/205923445-75f8b856-29f3-4847-bfc6-b84def76d57b.png">

## Ubuntu

```shell
$ curl -sL https://deb.nodesource.com/setup_18.x — Node.js 18 LTS "Hydrogen" | sudo bash -
$ sudo apt -y install nodejs
```

<!-- More -->

***

# Install Hexo & Theme

~~~applescript
sudo npm install hexo-cli -g
~~~

<img width="762" alt="Reinstall Hexo" src="/images/hexo-next-blog/205924373-26b84fb1-cf9b-438a-a2a6-84e1ce6df01a.png">

~~~bash
$ hexo init ${Blog_Dir_Name}
$ cd ${Blog_Dir_Name}
$ git clone https://github.com/next-theme/hexo-theme-next themes/next
~~~

***

# Configuration

~~날려버린 나의 commit들,,,~~

## Blog

1. [Update _config.yml](https://github.com/Zerohertz/Blog_Backup/commit/210c131974c50a7711cfb675350350b0fab540a5)
2. [Codeblock highlight](https://github.com/Zerohertz/Blog_Backup/commit/c117f92d440eec381c349111f2897c032cf494e8)
3. [Add posts](https://github.com/Zerohertz/Blog_Backup/commit/6dae7ebffe862fa18eaadca1f10182b58820f4d2)

## Theme

1. [Update _config.yml](https://github.com/Zerohertz/hexo-theme-next/commit/7d1444dd5d3a129483635a625be5502085bc298b)
2. [Anime duration down](https://github.com/Zerohertz/hexo-theme-next/commit/d4aa83a48c0ba2760401a07667cfcebc1784e1d6)
3. [Naver analytics](https://github.com/Zerohertz/hexo-theme-next/commit/6f5026c47fe4d66c88411f182804e7bb44c95546)
4. [Change achive title](https://github.com/Zerohertz/hexo-theme-next/commit/7043e819f46918b1120cbfc338e4e1acbf11f256)
5. [Move header](https://github.com/Zerohertz/hexo-theme-next/commit/0d1e1a69bcee9403a3e37d3dde9e7d380a4547dc)
6. [Resize mid_size of paginator](https://github.com/Zerohertz/hexo-theme-next/commit/3778268ebc4b95787e56c6fede3925aa897f9be3)
7. [Edit colors](https://github.com/Zerohertz/hexo-theme-next/commit/18f097f0e74f0baf8f0a23083414e8bfaf623f56)

## Dependencies

~~~bash
$ npm install hexo-deployer-git --save
$ npm install hexo-generator-searchdb --save
$ npm install hexo-related-posts --save
$ npm install hexo-generator-seo-friendly-sitemap --save
$ npm install hexo-generator-feed --save
$ npm install hexo-autonofollow --save
$ npm install hexo-auto-canonical --save
$ npm install hexo-generator-robotstxt --save
$ npm install hexo-word-counter --save
$ npm install @heowc/hexo-tag-gdemo --save
~~~

***

# 유용한 글

[skyksit](https://skyksit.com/categories/hexo/)
[SEO](https://alleyful.github.io/2019/08/10/tools/hexo/hexo-guide-03/)