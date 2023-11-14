import numpy as np
import torch

import math
import gym
import argparse
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import utils
import TD3

import ot
from scipy.spatial.distance import cdist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_ot(traj1, traj2, a_ot, b_ot, lamda1, max_iter, tcot=False, ot_cpu=False):
    # M = cdist(traj1, traj2, metric='sqeuclidean')
    M = np.sqrt(cdist(traj1, traj2, metric='euclidean'))
    # M = cdist(traj1, traj2, metric='cosine')
    M_t = M.copy()
    t_diff = np.zeros(M_t.shape)

    s_dim = traj1.shape[1]
    n = len(traj1)
    m = len(traj2)
    if tcot:
        for j in range(n):
            for k in range(m):
                M_t[j, k] = M_t[j, k]*(1.+abs(j/float(n)-k/float(m)))
                t_diff[j, k] = abs(j/float(n)-k/float(m))
    if not ot_cpu:
        M_t = torch.FloatTensor(M_t).to(device)
    T = -ot.bregman.sinkhorn_log(a_ot, b_ot, M_t, reg=lamda1, numItermax=max_iter, warn=True)
    # T = -ot.smooth.smooth_ot_dual(a_ot, b_ot, M_t, lamda1, reg_type='kl', numItermax=max_iter)
    # T = -ot.emd(a_ot, b_ot, M_t, numItermax=max_iter, numThreads='max')
    t_cost = M_t*T
    new_neg_ot = t_cost.sum(dim=0).cpu().numpy() if not ot_cpu else t_cost.sum(axis=0)
    if not ot_cpu:
        T = T.cpu().numpy()

    return new_neg_ot, -T, t_diff

# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, expert_dataset, demo_traj, filename, j, lamda1, eval_episodes=10, ot_cpu=True, use_time_index=False, collect_traj=False):
    avg_reward = 0.
    avg_neg_dist = 0.
    trajs = []
    if expert_dataset is not None:
        if not ot_cpu:
            a_ot, b_ot = torch.ones((len(demo_traj),)) / len(demo_traj), torch.ones((len(demo_traj),)) / len(demo_traj)  # uniform distribution on samples
            a_ot = a_ot.to(device)
            b_ot = b_ot.to(device)
        else:
            a_ot, b_ot = np.ones((len(demo_traj),)) / len(demo_traj), np.ones((len(demo_traj),)) / len(demo_traj)  # uniform distribution on samples

    for _ in range(eval_episodes):
        obs = env.reset()
        if expert_dataset is not None:
            obs_normed = expert_dataset.normz(obs)
        else:
            obs_normed = obs
        traj = []
        snsra = []
        done = False
        i = 0
        ep_rwd = 0.
        while i!=env._max_episode_steps:
            t_to_horizon = env._max_episode_steps - i
            action = policy.select_action(np.array(obs), t_to_horizon)
            new_obs, reward, done, _ = env.step(action)
            snsra.append([obs, action, new_obs, reward])
            if expert_dataset is not None:
                new_obs_normed = expert_dataset.normz(new_obs)
                action_normed = expert_dataset.normz(action, diff=True)
            else:
                new_obs_normed = new_obs
                action_normed = action
            obs_normed_t = obs_normed
            new_obs_normed_t = new_obs_normed
            if use_time_index:
                t = (t_to_horizon)/env._max_episode_steps
                t_next = (t_to_horizon - 1.)/env._max_episode_steps
                obs_normed_t = np.concatenate((obs_normed, [2.*t]), 0)
                new_obs_normed_t = np.concatenate((new_obs_normed, [2.*t_next]), 0)
            if state_action:
                traj.append(np.concatenate((obs_normed_t, action_normed), -1))
            else:
                traj.append(np.concatenate((obs_normed_t, new_obs_normed_t), -1))
            if done: reward *= 0.
            avg_reward += reward
            i+=1
            ep_rwd += reward
            obs = new_obs
            obs_normed = new_obs_normed
        trajs.append(snsra)
        if state_action:
            traj_np = np.array(traj).reshape((-1, state_dim+action_dim)) if not use_time_index else np.array(traj).reshape((-1, state_dim+action_dim+2))
        else:
            traj_np = np.array(traj).reshape((-1, 2*state_dim)) if not use_time_index else np.array(traj).reshape((-1, 2*state_dim+2))
        if demo_traj is not None:
            new_neg_ot, T, _ = compute_ot(demo_traj, traj_np, a_ot, b_ot, lamda1, max_iter, ot_cpu=ot_cpu, tcot=args.use_tcot)
            avg_neg_dist += new_neg_ot.sum()
        else:
            avg_neg_dist += 0.
    avg_reward /= eval_episodes
    avg_neg_dist /= eval_episodes

    if collect_traj:
        np.save('./trajs/'+file_name+'.npy', np.array(trajs))
    print("---------------------------------------")
    print("Evaluation over %d episodes: %f, %f" % (eval_episodes, avg_reward, avg_neg_dist))
    print("---------------------------------------")
    return avg_reward, avg_neg_dist


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="TD3")                    # Policy name
    parser.add_argument("--env_name", default="FetchPickAndPlaceDense-v1")            # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)                    # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=10e3, type=int)        # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)            # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=float)        # Max time steps to run environment for
    parser.add_argument("--save_models", action="store_true")            # Whether or not models are saved
    parser.add_argument("--expl_noise", default=0.2, type=float)        # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=100, type=int)            # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)            # Discount factor
    parser.add_argument("--n_demos", default=1, type=int)
    parser.add_argument("--tau", default=0.003, type=float)                # Target network update rate
    parser.add_argument("--policy_noise", default=0.1, type=float)        # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)        # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)            # Frequency of delayed policy updates
    parser.add_argument("--experts_path", default="./expert_trajectoriesHalfCheetah-v2_150.npy")
    parser.add_argument("--actor_lr", default=3e-4, type=float)
    parser.add_argument("--critic_lr", default=3e-4, type=float)
    parser.add_argument("--lamda1", default=.05, type=float)
    parser.add_argument("--lamda2", default=1., type=float)
    parser.add_argument("--delta", default=0.1, type=float)
    parser.add_argument("--actor_clip", default=0., type=float)
    parser.add_argument("--critic_clip", default=25., type=float)
    parser.add_argument("--max_iter", default=20000, type=int)
    parser.add_argument("--pol_updates", default=1000, type=int)
    parser.add_argument("--normalize", action="store_true")            # Whether or not models are saved
    parser.add_argument("--ot_cpu", action="store_true")
    parser.add_argument("--use_match", action="store_true")
    parser.add_argument("--aug_time", action="store_false")
    parser.add_argument("--use_time_index", action="store_true")
    parser.add_argument("--reward_scale", default=5., type=float)
    parser.add_argument("--std_clip", default=.33, type=float)
    parser.add_argument("--clip_state", default=False, action="store_false")
    parser.add_argument("--use_tcot", default=False, action="store_true")
    parser.add_argument("--dummy_bool", default=False, action="store_true")
    parser.add_argument("--state_action", default=False, action="store_true")


    args = parser.parse_args()

    file_name = "PAL_sqrtE_sklog_.5aug_%s_.2exp_Nexp5.scldrwd_Nrelpol_.33rdmMucbdS_1.Estdclip_3s.%sstdNclp%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_A256.%s.C1.1024.%s._%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s" % (str(args.discount), args.std_clip, args.policy_name, args.env_name, str(args.n_demos), str(args.max_iter), str(args.lamda1), str(args.lamda2), str(args.delta), str(args.tau), str(args.batch_size), str(args.reward_scale),str(args.policy_noise), str(args.actor_clip), str(args.critic_clip), str(args.actor_lr), str(args.critic_lr), str(args.pol_updates), args.ot_cpu, args.use_time_index, args.use_match, args.aug_time, args.clip_state, args.use_tcot, args.use_tcot, str(args.seed))
    print("---------------------------------------")
    print("Settings: %s" % (file_name))
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")
    if args.save_models and not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")

    n_demos = args.n_demos

    ot_cpu = args.ot_cpu
    use_time_index = args.use_time_index
    state_action = args.state_action

    demos = np.load(args.experts_path, allow_pickle=True)

    demo_state = np.vstack([np.stack(demos[i][:, 0]) for i in range(n_demos)])
    if state_action:
        demo_next_state = np.vstack([np.stack(demos[i][:, 1]) for i in range(n_demos)])
    else:
        demo_next_state = np.vstack([np.stack(demos[i][:, 3]) for i in range(n_demos)])
    demo_traj = np.concatenate((demo_state, demo_next_state), -1)


    env = gym.make(args.env_name)

    # Just for robotics environments
    env_max_episode_steps = env._max_episode_steps

    lamda1 = args.lamda1
    lamda2 = args.lamda2
    delta = args.delta
    max_iter = args.max_iter
    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])



    demo_traj = np.concatenate((demo_state, demo_next_state), -1)
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy
    if args.policy_name == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        kwargs["reward_scale"] = args.reward_scale
        kwargs["actor_lr"] = args.actor_lr
        kwargs["critic_lr"] = args.critic_lr
        kwargs["use_match"] = args.use_match
        kwargs["args"] = args

        policy = TD3.TD3(**kwargs)

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, 1)
    temp_buffer = utils.ReplayBuffer(state_dim, action_dim, 1)

    # Evaluate untrained policy
    evaluations = [evaluate_policy(policy, None, None, file_name, 0, lamda1, use_time_index=use_time_index, ot_cpu=ot_cpu)]

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    last_reward = 0.
    cur_traj = []

    first = True
    done_bool = True


    while total_timesteps < args.max_timesteps:
        if done_bool:
            if not first and len(cur_traj)==len(demo_traj):
                demo_traj = demo_trajs[np.random.randint(0, n_demos)]

                new_neg_ot, T, t_diff = compute_ot(demo_traj, cur_traj_np, a_ot, b_ot, lamda1, max_iter, tcot=args.use_tcot, ot_cpu=ot_cpu)
                matching = T.argmax(axis=0)

                tb_ptr = temp_buffer.ptr
                rb_ptr = replay_buffer.ptr

                for i in range(tb_ptr-rb_ptr):
                    s, a, ns, r, d, t, _ = temp_buffer.sample(1, rb_ptr+i)
                    replay_buffer.add(s.cpu().numpy(), a.cpu().numpy(), ns.cpu().numpy(), new_neg_ot[i], d.cpu().numpy(), t.cpu().numpy(), matching[i])
                episode_fake_reward += new_neg_ot.sum()

            if total_timesteps > args.start_timesteps and not first:

                ptr = replay_buffer.ptr
                s_rb = replay_buffer.state[:ptr]

                if args.policy_name == "TD3":
                    policy.train(replay_buffer, args.batch_size, iterations=args.pol_updates)
                else:
                    policy.train(replay_buffer, args.batch_size, iterations=args.pol_updates)
            if total_timesteps != 0:
                print(("Total T: %d Episode Num: %d Episode T: %d Reward: %f Fake Reward: %f") % (total_timesteps, episode_num, episode_timesteps, episode_reward, episode_fake_reward))
            # Evaluate episode
            if timesteps_since_eval >= args.eval_freq:
                timesteps_since_eval %= args.eval_freq
                if not first:
                    evaluations.append(evaluate_policy(policy, expert_dataset, demo_trajs[0], file_name, total_timesteps, lamda1, use_time_index=use_time_index, ot_cpu=ot_cpu))
                else:
                    evaluations.append(evaluate_policy(policy, None, None, file_name, total_timesteps, lamda1, use_time_index=use_time_index, ot_cpu=ot_cpu))
                if args.save_models: policy.save(file_name, directory="./pytorch_models")
                np.save("./results/%s" % (file_name), evaluations)

            # Reset environment
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_progress_reward = 0
            episode_fake_reward = 0
            episode_timesteps = 0
            episode_num += 1
            if not first:
                obs_normed = expert_dataset.normz(obs)
            else:
                obs_normed = obs

            cur_traj = []

            if first:
                neg_ot = 0.
            fake_reward = neg_ot

        t_to_horizon = env._max_episode_steps - episode_timesteps
        # Select action randomly or according to policy
        if total_timesteps < args.start_timesteps:
            action = env.action_space.sample()
        else:
            if first:
                ptr = replay_buffer.ptr
                s_rb = replay_buffer.state[:ptr]
                ns_rb = replay_buffer.next_state[:ptr]
                a_rb = replay_buffer.action[:ptr]
                r_rb = replay_buffer.reward[:ptr]

                n_trajs = int(ptr/env._max_episode_steps)

                rb_sar = np.hstack((s_rb, a_rb, r_rb, ns_rb))
                rb_sar = rb_sar.reshape((n_trajs, env._max_episode_steps, -1))
                reshaped_sar = [[[timestep[0:state_dim], timestep[state_dim:state_dim+action_dim], timestep[state_dim+action_dim], timestep[-state_dim]] for timestep in traj] for traj in rb_sar]
                np_sar = np.array(reshaped_sar)
                combined_eps = np.concatenate((demos[:n_demos].repeat(len(s_rb)//len(demo_state), axis=0), np_sar), axis=0)
                combined_dataset = utils.ExpertDataset(combined_eps, normalize=False)
                expert_dataset = utils.ExpertDataset(demos[:n_demos], normalize=False, clip=args.clip_state)
                expert_dataset.std = np.where(np.isclose(expert_dataset.std, 0.), 1., expert_dataset.std)
                expert_dataset.std = expert_dataset.std.clip(min=args.std_clip)
                demo_state = expert_dataset.normz(demo_state)
                demo_next_state = expert_dataset.normz(demo_next_state, diff=state_action)
                if use_time_index:
                    t = (np.arange(1000, 0, -1)/env._max_episode_steps).reshape((-1, 1))
                    t_next = (np.arange(999, -1, -1)/env._max_episode_steps).reshape((-1, 1))
                    demo_state = np.concatenate((demo_state, 2.*t), -1)
                    demo_next_state = np.concatenate((demo_next_state, 2.*t_next), -1)

                demo_trajs = np.concatenate((demo_state, demo_next_state), -1)
                if state_action:
                    demo_trajs = demo_trajs.reshape((n_demos, env._max_episode_steps, state_dim+action_dim))
                else:
                    demo_trajs = demo_trajs.reshape((n_demos, env._max_episode_steps, 2*state_dim))
                demo_traj = demo_trajs[0]
                rb_new_rewards = []
                rb_new_matchings = []
                # Reconstruct the rb
                if not ot_cpu:
                    a_ot, b_ot = torch.ones((len(demo_traj),)) / len(demo_traj), torch.ones((len(demo_traj),)) / len(demo_traj)  # uniform distribution on samples
                    a_ot = a_ot.to(device)
                    b_ot = b_ot.to(device)
                else:
                    a_ot, b_ot = np.ones((len(demo_traj),)) / len(demo_traj), np.ones((len(demo_traj),)) / len(demo_traj)  # uniform distribution on samples

                for i in range(ptr):
                    if i%env._max_episode_steps == 0:
                        rb_traj = []
                    s, a, ns, r, d, t, _ = replay_buffer.sample(1, i)
                    s_normed = expert_dataset.normz(s.cpu().numpy())
                    ns_normed = expert_dataset.normz(ns.cpu().numpy())
                    a_normed = expert_dataset.normz(a.cpu().numpy(), diff=True)
                    s_normed_t = s_normed
                    ns_normed_t = ns_normed

                    if use_time_index:
                        t = (env._max_episode_steps - i%env._max_episode_steps)/env._max_episode_steps
                        t_next = (env._max_episode_steps - i%env._max_episode_steps - 1.)/env._max_episode_steps
                        s_normed_t = np.concatenate((s_normed, [2.*t]), 0)
                        ns_normed_t = np.concatenate((ns_normed, [2.*t_next]), 0)

                    if state_action:
                        rb_traj.append(np.concatenate((s_normed_t, a_normed), -1))
                    else:
                        rb_traj.append(np.concatenate((s_normed_t, ns_normed_t), -1))
                    if state_action:
                        rb_traj_np = np.array(rb_traj).reshape((-1, state_dim+action_dim)) if not use_time_index else np.array(rb_traj).reshape((-1, state_dim+action_dim+2))
                    else:
                        rb_traj_np = np.array(rb_traj).reshape((-1, 2*state_dim)) if not use_time_index else np.array(rb_traj).reshape((-1, 2*state_dim+2))
                    if len(rb_traj) == 1000:
                        new_neg_ot, T, t_diff = compute_ot(demo_traj, rb_traj_np, a_ot, b_ot, lamda1, max_iter, tcot=args.use_tcot, ot_cpu=ot_cpu)
                        t_diff = (t_diff*T*1000.).sum(axis=0)
                        matching = T.argmax(axis=0)

                        rb_new_rewards.append(new_neg_ot)
                        rb_new_matchings.append(matching)
                        demo_traj = demo_trajs[np.random.randint(0, n_demos)]
                rb_new_rewards = np.vstack(rb_new_rewards)
                rb_new_matchings = np.vstack(rb_new_matchings)
                replay_buffer.match[:replay_buffer.ptr] = np.reshape(rb_new_matchings, (-1, 1))
                replay_buffer.reward[:replay_buffer.ptr] = np.reshape(rb_new_rewards, (-1, 1))

                rdm_mean = np.mean(s_rb, axis=0)
                rdm_std = np.std(s_rb, axis=0)
                rdm_std = np.where(np.isclose(rdm_std, 0.), 1., rdm_std)

                combined_std = combined_dataset.std
                combined_std = np.where(np.isclose(combined_std, 0.), 1., combined_std)
                combined_std = combined_std.clip(min=.33)
                print(rdm_mean.shape)
                print(combined_std)
                policy.set_mean_std(rdm_mean, combined_std)
                first = False

            action = policy.select_action(np.array(obs), t_to_horizon)
            if args.expl_noise != 0:
                action = (action + np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)

        # Perform action
        new_obs, reward, done, _ = env.step(action)
        if done: reward *= 0.
        if not first:
            new_obs_normed = expert_dataset.normz(new_obs)
            action_normed = expert_dataset.normz(action, diff=True)
        else:
            new_obs_normed = new_obs
            action_normed = action
        obs_normed_t = obs_normed
        new_obs_normed_t = new_obs_normed

        if use_time_index:
            t = t_to_horizon/env._max_episode_steps
            t_next = (t_to_horizon-1.)/env._max_episode_steps
            obs_normed_t = np.concatenate((obs_normed, [2.*t]), 0)
            new_obs_normed_t = np.concatenate((new_obs_normed, [2.*t_next]), 0)
        if state_action:
            cur_traj.append(np.concatenate((obs_normed_t, action_normed), -1))
        else:
            cur_traj.append(np.concatenate((obs_normed_t, new_obs_normed_t), -1))
        if state_action:
            cur_traj_np = np.array(cur_traj).reshape((-1, state_dim+action_dim)) if not use_time_index else np.array(cur_traj).reshape((-1, state_dim+action_dim+2))
        else:
            cur_traj_np = np.array(cur_traj).reshape((-1, 2*state_dim)) if not use_time_index else np.array(cur_traj).reshape((-1, 2*state_dim+2))


        # if first:
        new_neg_ot = reward # placeholder
        # compute_opw
        fake_reward = new_neg_ot
        done_bool = float(done) if episode_timesteps == env._max_episode_steps - 1 else 0

        episode_reward += reward

        # Store data in replay buffer
        if not first:
            temp_buffer.add(obs, action, new_obs, fake_reward, done_bool, t_to_horizon, -1)
        else:
            replay_buffer.add(obs, action, new_obs, fake_reward, done_bool, t_to_horizon, -1)
            temp_buffer.add(obs, action, new_obs, fake_reward, done_bool, t_to_horizon, -1)


        obs = new_obs
        obs_normed = new_obs_normed
        neg_ot = new_neg_ot

        total_timesteps += 1
        episode_timesteps += 1
        timesteps_since_eval += 1

        last_reward = reward

    # Final evaluation
    evaluations.append(evaluate_policy(policy, expert_dataset, demo_trajs[0], file_name, total_timesteps, lamda1, use_time_index=use_time_index, ot_cpu=ot_cpu, collect_traj=True))
    np.save("./results/%s" % (file_name), evaluations)
    np.save("./trajs/"+file_name+'_mustd.npy', np.array([expert_dataset.mean, expert_dataset.std]))
    if args.save_models: policy.save("%s" % (file_name), directory="./pytorch_models")
